package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LiveExposure
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** A typed slot in [PipelineContext] that one stage fills and later stages read. */
class PipelineKey<T : Any>(val name: String) {
    override fun toString() = name
}

/** Keys shared by the built-in stages. A new stage may define its own. */
object PipelineKeys {
    /** Small sRGB preview in the sensor's orientation (what DNG thumbnails expect). */
    val QuickLook = PipelineKey<Bitmap>("quickLook")

    /** The same preview rotated upright, for the viewfinder's thumbnail. */
    val QuickLookUpright = PipelineKey<Bitmap>("quickLookUpright")

    /** The native burst session shared by align, merge and finish. */
    val Native = PipelineKey<NativeBurst>("native")

    /** The finished full-resolution image in the sensor's orientation. */
    val Rendered = PipelineKey<Bitmap>("rendered")

    /** The same image as an Ultra HDR JPEG file (SDR base + gain map), when the run asked for it. */
    val UltraHdrJpeg = PipelineKey<ByteArray>("ultraHdrJpeg")
}

/** Working state of one pipeline run: the burst, intermediate results, and what was written. */
class PipelineContext(
    val burst: RawBurst,
    val sink: OutputSink,
) {
    private val slots = HashMap<PipelineKey<*>, Any>()
    private val _outputs = mutableListOf<SavedOutput>()
    private val resources = mutableListOf<AutoCloseable>()

    val outputs: List<SavedOutput> get() = _outputs

    operator fun <T : Any> set(key: PipelineKey<T>, value: T) {
        slots[key] = value
    }

    @Suppress("UNCHECKED_CAST")
    operator fun <T : Any> get(key: PipelineKey<T>): T? = slots[key] as T?

    fun <T : Any> require(key: PipelineKey<T>): T = checkNotNull(get(key)) { "No $key: a stage before this one has to produce it" }

    fun addOutput(output: SavedOutput) {
        _outputs += output
    }

    /** Closes [resource] when the run ends, successfully or not. */
    fun <T : AutoCloseable> own(resource: T): T = resource.also { resources += it }

    internal fun closeResources() {
        resources.asReversed().forEach { runCatching { it.close() } }
        resources.clear()
    }
}

/**
 * One step of processing: quick look, align, merge, finish, JPEG / DNG writing. Stages read and
 * write [PipelineContext] slots.
 */
interface PipelineStage {
    val name: String

    suspend fun run(ctx: PipelineContext)
}

data class StageTiming(val stage: String, val millis: Long)

data class PipelineResult(
    val outputs: List<SavedOutput>,
    val thumbnail: Bitmap?,
    val timings: List<StageTiming>,
) {
    /** What the thumbnail button opens: the rendered photo when there is one, else the DNG. */
    val primaryUri: Uri? get() = (outputs.firstOrNull { it.kind != OutputKind.DNG } ?: outputs.firstOrNull())?.uri

    val totalMillis: Long get() = timings.sumOf { it.millis }
}

interface ImagePipeline {
    val id: String

    /** RAW frames to capture per shutter press. */
    val framesPerCapture: Int get() = 1

    /** The exposure to capture with, given what the user asked for and what the preview's AE measured. */
    fun exposureFor(requested: ExposurePlan, live: LiveExposure, camera: CameraInfo): ExposurePlan = requested

    suspend fun process(burst: RawBurst, sink: OutputSink, progress: (Float) -> Unit = {}): PipelineResult
}

/** Runs its stages in order on a background dispatcher, reporting progress per stage. */
open class StagedPipeline(
    override val id: String,
    private val stages: List<PipelineStage>,
) : ImagePipeline {
    override suspend fun process(burst: RawBurst, sink: OutputSink, progress: (Float) -> Unit): PipelineResult =
        withContext(Dispatchers.Default) {
            val ctx = PipelineContext(burst, sink)
            val timings = ArrayList<StageTiming>(stages.size)
            try {
                stages.forEachIndexed { i, stage ->
                    val start = SystemClock.elapsedRealtime()
                    stage.run(ctx)
                    timings += StageTiming(stage.name, SystemClock.elapsedRealtime() - start)
                    progress((i + 1f) / stages.size)
                }
            } finally {
                ctx.closeResources()
            }
            Log.i(TAG, "$id ${burst.size} frames: " + timings.joinToString { "${it.stage} ${it.millis} ms" } + ", total ${timings.sumOf { it.millis }} ms")
            PipelineResult(ctx.outputs, ctx[PipelineKeys.QuickLookUpright], timings)
        }

    private companion object {
        const val TAG = "PocoPawPipeline"
    }
}

/** What the burst is captured for: one exposure (auto), an exposure bracket, or long night frames. */
enum class BurstScene { AUTO, BRACKET, NIGHT }

/** How a pipeline was asked to run: frames per press and which files to write. */
data class PipelineOptions(
    val frames: Int = HdrPlusPipeline.DEFAULT_FRAMES,
    val writeJpeg: Boolean = true,
    val writeDng: Boolean = false,
    val look: BurstLook = BurstLook(),
    val scene: BurstScene = BurstScene.AUTO,
    val ultraHdr: Boolean = false,
    /** Lateral CA of the capturing lens, applied by the kernel-regression merge. */
    val lca: LensLca? = null,
    /** Calibrated noise of the capturing camera; null estimates it from each burst. */
    val noise: NoiseCurve? = null,
    /** Run the native steps on the GPU where possible ([NativeBurst.useGpu]). */
    val gpu: Boolean = false,
)

/** The single-frame pipeline: quick look, then the reference frame written as DNG unchanged. */
class DngPassthroughPipeline : StagedPipeline(ID, listOf(QuickLookStage(), DngWriteStage())) {
    companion object {
        const val ID = "dng_passthrough"
    }
}

/**
 * The multi-frame CPU pipeline of `docs/research/PIPELINE_REPORT.md`: HDR+ tile alignment, then
 * either the spatial robust merge plus Malvar-He-Cutler demosaic ([kernelRegression] false) or the
 * handheld super-resolution kernel-regression merge at 1x (true), then LSC / WB, CCM, chroma
 * denoise, synthetic-exposure fusion, tone curve, sharpening and JPEG. With [PipelineOptions.writeDng]
 * the reference frame is also saved as DNG, before the merge so the RAW is safe early.
 * [BurstScene.BRACKET] captures two short frames per long one (merged on the short frame's scale);
 * [BurstScene.NIGHT] meters a motion-limited shutter, merges in the frequency domain (classic path)
 * and finishes with the night look.
 */
class HdrPlusPipeline(
    id: String,
    kernelRegression: Boolean,
    private val options: PipelineOptions,
) : StagedPipeline(id, stagesFor(kernelRegression, options)) {

    override val framesPerCapture: Int get() = options.frames

    override fun exposureFor(requested: ExposurePlan, live: LiveExposure, camera: CameraInfo): ExposurePlan =
        when (requested) {
            is ExposurePlan.Auto -> if (options.frames <= 1) requested else when (options.scene) {
                BurstScene.AUTO -> BurstExposure.plan(live, camera, requested.evIndex)
                BurstScene.BRACKET -> BurstExposure.bracket(live, camera, requested.evIndex)
                BurstScene.NIGHT -> BurstExposure.night(live, camera, requested.evIndex)
            } ?: requested
            is ExposurePlan.Manual, is ExposurePlan.PerFrame -> requested
        }

    companion object {
        const val ID = "hdrplus"
        const val ID_KR = "hdrplus_kr"
        const val DEFAULT_FRAMES = 8

        private fun stagesFor(kernelRegression: Boolean, o: PipelineOptions): List<PipelineStage> = buildList {
            if (o.writeDng) {
                add(QuickLookStage())
                add(DngWriteStage())
            }
            val night = o.scene == BurstScene.NIGHT
            add(AlignStage(o.gpu, o.noise))
            add(MergeStage(MergeOptions(kernelRegression = kernelRegression, frequencyMerge = night, lca = o.lca)))
            add(FinishStage(if (night) o.look.copy(style = "night") else o.look, o.ultraHdr))
            if (o.writeJpeg) add(JpegWriteStage())
        }
    }
}

/**
 * Every pipeline the app can run, by id. The camera screen picks one through settings, without
 * changes to the camera layer.
 */
object PipelineRegistry {
    private val factories = linkedMapOf<String, (PipelineOptions) -> ImagePipeline>(
        DngPassthroughPipeline.ID to { _ -> DngPassthroughPipeline() },
        HdrPlusPipeline.ID to { o -> HdrPlusPipeline(HdrPlusPipeline.ID, kernelRegression = false, o) },
        HdrPlusPipeline.ID_KR to { o -> HdrPlusPipeline(HdrPlusPipeline.ID_KR, kernelRegression = true, o) },
    )

    val ids: List<String> get() = factories.keys.toList()

    val defaultId: String = HdrPlusPipeline.ID

    fun register(id: String, factory: (PipelineOptions) -> ImagePipeline) {
        factories[id] = factory
    }

    fun create(id: String?, options: PipelineOptions = PipelineOptions()): ImagePipeline =
        (factories[id] ?: factories.getValue(defaultId)).invoke(options)
}
