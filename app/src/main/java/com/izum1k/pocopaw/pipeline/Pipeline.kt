package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
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
}

/** Working state of one pipeline run: the burst, intermediate results, and what was written. */
class PipelineContext(
    val burst: RawBurst,
    val sink: OutputSink,
) {
    private val slots = HashMap<PipelineKey<*>, Any>()
    private val _outputs = mutableListOf<SavedOutput>()

    val outputs: List<SavedOutput> get() = _outputs

    operator fun <T : Any> set(key: PipelineKey<T>, value: T) {
        slots[key] = value
    }

    @Suppress("UNCHECKED_CAST")
    operator fun <T : Any> get(key: PipelineKey<T>): T? = slots[key] as T?

    fun addOutput(output: SavedOutput) {
        _outputs += output
    }
}

/**
 * One step of processing: quick look, DNG writing today; later alignment, merge, demosaic,
 * tone mapping, JPEG / Ultra HDR encoding. Stages read and write [PipelineContext] slots.
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
    val primaryUri: Uri? get() = outputs.firstOrNull()?.uri
}

interface ImagePipeline {
    val id: String

    /** RAW frames to capture per shutter press. */
    val framesPerCapture: Int get() = 1

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
            stages.forEachIndexed { i, stage ->
                val start = SystemClock.elapsedRealtime()
                stage.run(ctx)
                timings += StageTiming(stage.name, SystemClock.elapsedRealtime() - start)
                progress((i + 1f) / stages.size)
            }
            PipelineResult(ctx.outputs, ctx[PipelineKeys.QuickLookUpright], timings)
        }
}

/** The single-frame pipeline: quick look, then the reference frame written as DNG unchanged. */
class DngPassthroughPipeline : StagedPipeline(ID, listOf(QuickLookStage(), DngWriteStage())) {
    companion object {
        const val ID = "dng_passthrough"
    }
}

/**
 * Every pipeline the app can run, by id. A multi-frame pipeline registers here (for example
 * `StagedPipeline("hdrplus", listOf(AlignStage(), MergeStage(), ..., DngWriteStage()))`) and the
 * camera screen picks it up through settings, without changes to the camera layer.
 */
object PipelineRegistry {
    private val factories = linkedMapOf<String, () -> ImagePipeline>(
        DngPassthroughPipeline.ID to { DngPassthroughPipeline() },
    )

    val ids: List<String> get() = factories.keys.toList()

    val defaultId: String = DngPassthroughPipeline.ID

    fun register(id: String, factory: () -> ImagePipeline) {
        factories[id] = factory
    }

    fun create(id: String?): ImagePipeline = (factories[id] ?: factories.getValue(defaultId)).invoke()
}
