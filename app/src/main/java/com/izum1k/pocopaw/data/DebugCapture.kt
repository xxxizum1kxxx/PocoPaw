package com.izum1k.pocopaw.data

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import android.os.Build
import android.util.Log
import android.util.Size
import android.util.SizeF
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.pipeline.BurstStream
import com.izum1k.pocopaw.pipeline.ImagePipeline
import com.izum1k.pocopaw.pipeline.OutputSink
import com.izum1k.pocopaw.pipeline.PipelineResult
import com.izum1k.pocopaw.pipeline.RawBurst
import com.izum1k.pocopaw.pipeline.RawFrame
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.roundToLong

@Serializable
data class RectJson(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val width: Int = right - left,
    val height: Int = bottom - top,
)

@Serializable
data class IntSizeJson(val width: Int, val height: Int)

@Serializable
data class FloatSizeJson(val width: Float, val height: Float)

@Serializable
data class IntRangeJson(val lower: Int, val upper: Int)

@Serializable
data class LongRangeJson(val lower: Long, val upper: Long)

@Serializable
data class ManualStepJson(val iso: Int, val exposureTimeNs: Long)

@Serializable
data class ExposurePlanJson(
    val mode: String,
    val evIndex: Int? = null,
    val manual: ManualStepJson? = null,
    val perFrame: List<ManualStepJson>? = null,
)

@Serializable
data class PreviewMeteringJson(
    val iso: Int? = null,
    val exposureTimeNs: Long? = null,
    val focusDioptres: Float? = null,
    val focusLocked: Boolean = false,
    val aeConverged: Boolean = false,
    val flashRequired: Boolean = false,
    val motionRadPerSec: Float? = null,
)

@Serializable
data class GainBreakdown(
    val analogGain: Double?,
    val sensorDigitalGain: Double,
    val postRawDigitalGain: Double,
    val totalDigitalGain: Double,
    val totalGain: Double?,
)

@Serializable
data class LensShadingJson(
    val rows: Int,
    val cols: Int,
    val gainsRggb: List<Float>,
)

@Serializable
data class NoiseProfileChannelJson(
    val scaleS: Double,
    val offsetO: Double,
)

@Serializable
data class StageTimingJson(
    val stage: String,
    val millis: Long,
)

@Serializable
data class SavedOutputJson(
    val kind: String,
    val displayName: String,
    val uri: String,
)

@Serializable
data class FrameDebugInfo(
    val index: Int,
    val isReference: Boolean = false,
    val file: String,
    val width: Int,
    val height: Int,
    val rowStride: Int,
    val byteLength: Int,
    val pixelEncoding: String = "RAW16_LE",
    val sensorTimestampNs: Long,
    val frameNumber: Long? = null,
    val physicalCameraId: String,
    val exposureTimeNs: Long? = null,
    val sensitivityIso: Int? = null,
    val postRawSensitivityBoost: Int? = null,
    val analogGain: Double? = null,
    val digitalGain: Double? = null,
    val totalGain: Double? = null,
    val frameDurationNs: Long? = null,
    val rollingShutterSkewNs: Long? = null,
    val blackLevel: List<Float>,
    val whiteLevel: Int,
    val awbMode: Int? = null,
    val awbState: Int? = null,
    val awbNeutral: List<Double>? = null,
    val awbGainsRggb: List<Float>? = null,
    val ccm: List<Double>? = null,
    val lensShadingMap: LensShadingJson? = null,
    val noiseProfile: List<NoiseProfileChannelJson>? = null,
    val cropRegion: RectJson? = null,
    val zoomRatio: Float? = null,
    val focalLengthMm: Float? = null,
    val apertureFNumber: Float? = null,
    val focusDistanceDioptres: Float? = null,
    val lensState: Int? = null,
    val afMode: Int? = null,
    val afState: Int? = null,
    val aeMode: Int? = null,
    val aeState: Int? = null,
    val flashMode: Int? = null,
    val flashState: Int? = null,
)

@Serializable
data class CaptureDebugManifest(
    val shotId: String,
    val captureTimeMillis: Long,
    val captureTimeIsoUtc: String,
    val packageName: String,
    val appVersion: String,
    val deviceManufacturer: String,
    val deviceModel: String,
    val deviceBoard: String,
    val cameraId: String,
    val facing: String,
    val cfaPattern: String,
    val cfaArrangement: Int? = null,
    val orientationDegrees: Int,
    val sensorOrientation: Int? = null,
    val zoomRatio: Float,
    val cropZoom: Float,
    val aspectRatio: String,
    val activeArray: RectJson? = null,
    val preCorrectionActiveArray: RectJson? = null,
    val pixelArraySize: IntSizeJson? = null,
    val physicalSizeMm: FloatSizeJson? = null,
    val staticBlackLevel: List<Int>? = null,
    val staticWhiteLevel: Int? = null,
    val maxAnalogSensitivity: Int? = null,
    val sensitivityRange: IntRangeJson? = null,
    val exposureTimeRangeNs: LongRangeJson? = null,
    val referenceIlluminant1: Int? = null,
    val referenceIlluminant2: Int? = null,
    val colorTransform1: List<Double>? = null,
    val colorTransform2: List<Double>? = null,
    val forwardMatrix1: List<Double>? = null,
    val forwardMatrix2: List<Double>? = null,
    val calibrationTransform1: List<Double>? = null,
    val calibrationTransform2: List<Double>? = null,
    val exposurePlan: ExposurePlanJson,
    val previewMetering: PreviewMeteringJson,
    val pipelineId: String,
    val frameCount: Int,
    val referenceFrameIndex: Int,
    val totalPipelineMs: Long? = null,
    val stageTimings: List<StageTimingJson> = emptyList(),
    val savedOutputs: List<SavedOutputJson> = emptyList(),
    val pipelineError: String? = null,
    val frames: List<FrameDebugInfo>,
)

object DebugCapture {
    const val SUBDIR = "debug_captures"
    const val MANIFEST_FILE = "capture_debug.json"
    private const val TAG = "PocoPawDebug"
    private val shotSeq = AtomicInteger(0)

    internal val json = Json {
        prettyPrint = true
        encodeDefaults = true
    }

    fun rootDir(context: Context): File =
        context.getExternalFilesDir(SUBDIR) ?: File(context.filesDir, SUBDIR)

    fun createShotDir(root: File, captureTimeMillis: Long, cameraId: String): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(captureTimeMillis))
        val seq = shotSeq.incrementAndGet()
        val safeCam = cameraId.replace(Regex("[^A-Za-z0-9_-]"), "_")
        val base = String.format(Locale.US, "%s_shot%04d_cam%s", stamp, seq, safeCam)
        var dir = File(root, base)
        var suffix = 2
        while (dir.exists()) {
            dir = File(root, "${base}_${suffix++}")
        }
        dir.mkdirs()
        return dir
    }

    fun rawFileName(index: Int): String = String.format(Locale.US, "frame_%02d.raw", index)

    /** Writes [buffer] from position 0 to [target] without heap copies; returns bytes written. */
    fun writeRawBuffer(target: File, buffer: ByteBuffer): Int {
        val dup = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        dup.rewind()
        val total = dup.remaining()
        FileOutputStream(target).use { out ->
            val ch = out.channel
            while (dup.hasRemaining()) {
                ch.write(dup)
            }
        }
        return total
    }

    /** Atomically writes `capture_debug.json` via a `.tmp` sibling so readers never see partial JSON. */
    fun writeManifest(shotDir: File, manifest: CaptureDebugManifest): File {
        shotDir.mkdirs()
        val finalFile = File(shotDir, MANIFEST_FILE)
        val tmpFile = File(shotDir, "$MANIFEST_FILE.tmp")
        tmpFile.writeText(json.encodeToString(manifest), Charsets.UTF_8)
        if (!tmpFile.renameTo(finalFile)) {
            tmpFile.copyTo(finalFile, overwrite = true)
            tmpFile.delete()
        }
        return finalFile
    }

    fun computeGains(iso: Int?, minIso: Int?, maxAnalogIso: Int?, postRawBoost: Int?): GainBreakdown {
        val sensorDigital = if (iso != null && maxAnalogIso != null && maxAnalogIso > 0 && iso > maxAnalogIso) {
            iso.toDouble() / maxAnalogIso
        } else {
            1.0
        }
        val postDigital = if (postRawBoost != null && postRawBoost > 0) postRawBoost / 100.0 else 1.0
        val totalDigital = round6(sensorDigital * postDigital)
        val analog = if (iso != null && minIso != null && minIso > 0) {
            val analogIso = if (maxAnalogIso != null && maxAnalogIso > 0) minOf(iso, maxAnalogIso) else iso
            round6(analogIso.toDouble() / minIso)
        } else {
            null
        }
        val total = analog?.let { round6(it * totalDigital) }
        return GainBreakdown(
            analogGain = analog,
            sensorDigitalGain = round6(sensorDigital),
            postRawDigitalGain = round6(postDigital),
            totalDigitalGain = totalDigital,
            totalGain = total,
        )
    }

    fun serializeExposurePlan(plan: ExposurePlan): ExposurePlanJson = when (plan) {
        is ExposurePlan.Auto -> ExposurePlanJson(mode = "Auto", evIndex = plan.evIndex)
        is ExposurePlan.Manual -> ExposurePlanJson(
            mode = "Manual",
            manual = ManualStepJson(plan.exposure.iso, plan.exposure.exposureNs),
        )
        is ExposurePlan.PerFrame -> ExposurePlanJson(
            mode = "PerFrame",
            perFrame = plan.frames.map { ManualStepJson(it.iso, it.exposureNs) },
        )
    }

    fun serializePreviewMetering(live: LiveExposure): PreviewMeteringJson = PreviewMeteringJson(
        iso = live.iso,
        exposureTimeNs = live.exposureNs,
        focusDioptres = live.focusDioptres?.finiteOrNull(),
        focusLocked = live.focusLocked,
        aeConverged = live.aeConverged,
        flashRequired = live.flashRequired,
        motionRadPerSec = live.motionRadPerSec?.finiteOrNull(),
    )

    fun formatIsoUtc(millis: Long): String =
        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }.format(Date(millis))

    internal fun round6(v: Double): Double =
        if (!v.isFinite()) 0.0 else (v * 1_000_000.0).roundToLong() / 1_000_000.0

    internal fun Float.finiteOrNull(): Float? = if (isFinite()) this else null
    internal fun Double.finiteOrNull(): Double? = if (isFinite()) this else null

    internal fun dumpRawAndSnapshot(
        shotDir: File,
        burst: RawBurst,
        packageName: String,
        versionName: String,
        aspect: AspectRatio,
        exposurePlan: ExposurePlan,
        liveExposure: LiveExposure,
        pipelineId: String,
    ): CaptureDebugManifest {
        val meta = burst.metadata
        val cc = meta.characteristics
        val minIso = cc.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.lower
        val maxAnalog = cc.get(CameraCharacteristics.SENSOR_MAX_ANALOG_SENSITIVITY)
        val staticPattern = cc.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
        val staticBlack = staticPattern?.let { p -> List(4) { i -> p.getOffsetForIndex(i % 2, i / 2) } }
        val staticWhite = cc.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL)

        val frameInfos = burst.frames.mapIndexed { idx, frame ->
            val fileName = rawFileName(idx)
            val byteLen = writeRawBuffer(File(shotDir, fileName), frame.data)
            extractFrameInfo(
                index = idx,
                fileName = fileName,
                byteLen = byteLen,
                frame = frame,
                fallbackCameraId = meta.cameraId,
                fallbackBlack = meta.blackLevel,
                fallbackWhite = meta.whiteLevel,
                minIso = minIso,
                maxAnalogIso = maxAnalog,
            )
        }

        return CaptureDebugManifest(
            shotId = shotDir.name,
            captureTimeMillis = meta.captureTimeMillis,
            captureTimeIsoUtc = formatIsoUtc(meta.captureTimeMillis),
            packageName = packageName,
            appVersion = versionName,
            deviceManufacturer = Build.MANUFACTURER.orEmpty(),
            deviceModel = Build.MODEL.orEmpty(),
            deviceBoard = Build.DEVICE.orEmpty(),
            cameraId = meta.cameraId,
            facing = meta.facing.name,
            cfaPattern = meta.cfa.name,
            cfaArrangement = cc.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT),
            orientationDegrees = meta.orientationDegrees,
            sensorOrientation = cc.get(CameraCharacteristics.SENSOR_ORIENTATION),
            zoomRatio = meta.zoomRatio.finiteOrNull() ?: 1f,
            cropZoom = meta.cropZoom.finiteOrNull() ?: 1f,
            aspectRatio = "${aspect.width}:${aspect.height}",
            activeArray = cc.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)?.toJson(),
            preCorrectionActiveArray = cc.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)?.toJson(),
            pixelArraySize = cc.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)?.toJson(),
            physicalSizeMm = cc.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.toJson(),
            staticBlackLevel = staticBlack,
            staticWhiteLevel = staticWhite,
            maxAnalogSensitivity = maxAnalog,
            sensitivityRange = cc.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)?.let { IntRangeJson(it.lower, it.upper) },
            exposureTimeRangeNs = cc.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)?.let { LongRangeJson(it.lower, it.upper) },
            referenceIlluminant1 = cc.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1),
            referenceIlluminant2 = cc.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2)?.let { it.toInt() and 0xFF },
            colorTransform1 = cc.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1)?.toMatrixList(),
            colorTransform2 = cc.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2)?.toMatrixList(),
            forwardMatrix1 = cc.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX1)?.toMatrixList(),
            forwardMatrix2 = cc.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX2)?.toMatrixList(),
            calibrationTransform1 = cc.get(CameraCharacteristics.SENSOR_CALIBRATION_TRANSFORM1)?.toMatrixList(),
            calibrationTransform2 = cc.get(CameraCharacteristics.SENSOR_CALIBRATION_TRANSFORM2)?.toMatrixList(),
            exposurePlan = serializeExposurePlan(exposurePlan),
            previewMetering = serializePreviewMetering(liveExposure),
            pipelineId = pipelineId,
            frameCount = burst.size,
            referenceFrameIndex = burst.referenceIndex,
            frames = frameInfos,
        )
    }

    private fun extractFrameInfo(
        index: Int,
        fileName: String,
        byteLen: Int,
        frame: RawFrame,
        fallbackCameraId: String,
        fallbackBlack: FloatArray,
        fallbackWhite: Int,
        minIso: Int?,
        maxAnalogIso: Int?,
    ): FrameDebugInfo {
        val r = frame.result
        val iso = r.get(CaptureResult.SENSOR_SENSITIVITY)
        val expNs = r.get(CaptureResult.SENSOR_EXPOSURE_TIME)
        val boost = r.get(CaptureResult.CONTROL_POST_RAW_SENSITIVITY_BOOST)
        val gains = computeGains(iso, minIso, maxAnalogIso, boost)
        val dynBlack = r.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
        val black = List(4) { i ->
            (dynBlack?.getOrNull(i) ?: fallbackBlack.getOrElse(i) { 0f }).finiteOrNull() ?: 0f
        }
        val white = r.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL) ?: fallbackWhite
        val wb = r.get(CaptureResult.COLOR_CORRECTION_GAINS)
        val neutral = r.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)
        val lsm = r.get(CaptureResult.STATISTICS_LENS_SHADING_CORRECTION_MAP)
        val lsmJson = lsm?.let { map ->
            val rawGains = FloatArray(map.gainFactorCount).also { map.copyGainFactors(it, 0) }
            LensShadingJson(
                rows = map.rowCount,
                cols = map.columnCount,
                gainsRggb = rawGains.map { v -> if (v.isFinite()) (v * 100_000f).roundToLong() / 100_000f else 1f },
            )
        }
        val noise = r.get(CaptureResult.SENSOR_NOISE_PROFILE)?.map { pair ->
            NoiseProfileChannelJson(
                scaleS = pair.first?.finiteOrNull() ?: 0.0,
                offsetO = pair.second?.finiteOrNull() ?: 0.0,
            )
        }
        val physicalId = r.get(CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID)
            ?.takeIf { it.isNotBlank() } ?: fallbackCameraId
        val stride = if (frame.height > 0 && byteLen > 0) byteLen / frame.height else frame.width * 2

        return FrameDebugInfo(
            index = index,
            file = fileName,
            width = frame.width,
            height = frame.height,
            rowStride = stride,
            byteLength = byteLen,
            sensorTimestampNs = r.get(CaptureResult.SENSOR_TIMESTAMP) ?: frame.timestampNs,
            frameNumber = runCatching { r.frameNumber }.getOrNull(),
            physicalCameraId = physicalId,
            exposureTimeNs = expNs,
            sensitivityIso = iso,
            postRawSensitivityBoost = boost,
            analogGain = gains.analogGain,
            digitalGain = gains.totalDigitalGain,
            totalGain = gains.totalGain,
            frameDurationNs = r.get(CaptureResult.SENSOR_FRAME_DURATION),
            rollingShutterSkewNs = r.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW),
            blackLevel = black,
            whiteLevel = white,
            awbMode = r.get(CaptureResult.CONTROL_AWB_MODE),
            awbState = r.get(CaptureResult.CONTROL_AWB_STATE),
            awbNeutral = neutral?.map { round6(it.toDouble()) },
            awbGainsRggb = wb?.let {
                listOf(
                    it.red.finiteOrNull() ?: 1f,
                    it.greenEven.finiteOrNull() ?: 1f,
                    it.greenOdd.finiteOrNull() ?: 1f,
                    it.blue.finiteOrNull() ?: 1f,
                )
            },
            ccm = r.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)?.toMatrixList(),
            lensShadingMap = lsmJson,
            noiseProfile = noise,
            cropRegion = r.get(CaptureResult.SCALER_CROP_REGION)?.toJson(),
            zoomRatio = r.get(CaptureResult.CONTROL_ZOOM_RATIO)?.finiteOrNull(),
            focalLengthMm = r.get(CaptureResult.LENS_FOCAL_LENGTH)?.finiteOrNull(),
            apertureFNumber = r.get(CaptureResult.LENS_APERTURE)?.finiteOrNull(),
            focusDistanceDioptres = r.get(CaptureResult.LENS_FOCUS_DISTANCE)?.finiteOrNull(),
            lensState = r.get(CaptureResult.LENS_STATE),
            afMode = r.get(CaptureResult.CONTROL_AF_MODE),
            afState = r.get(CaptureResult.CONTROL_AF_STATE),
            aeMode = r.get(CaptureResult.CONTROL_AE_MODE),
            aeState = r.get(CaptureResult.CONTROL_AE_STATE),
            flashMode = r.get(CaptureResult.FLASH_MODE),
            flashState = r.get(CaptureResult.FLASH_STATE),
        )
    }

    private fun Rect.toJson() = RectJson(left, top, right, bottom, width(), height())
    private fun Size.toJson() = IntSizeJson(width, height)
    private fun SizeF.toJson() = FloatSizeJson(width.finiteOrNull() ?: 0f, height.finiteOrNull() ?: 0f)

    private fun ColorSpaceTransform.toMatrixList(): List<Double> =
        List(9) { k -> round6(getElement(k % 3, k / 3).toDouble()) }

    internal fun logSaved(dir: File, frames: Int) {
        Log.i(TAG, "saved debug bundle ($frames frames) to ${dir.absolutePath}")
    }

    internal fun logFailed(e: Throwable) {
        Log.w(TAG, "debug capture bundle failed", e)
    }
}

/**
 * Wraps an [ImagePipeline] when Debug Data Capture is enabled: disables in-capture frame streaming
 * so [RawBurst.frames] stay intact in [com.izum1k.pocopaw.pipeline.NativeMemory], streams every raw
 * frame to disk, runs the delegate pipeline, and writes `capture_debug.json` with reference frame
 * index and per-stage timings before [com.izum1k.pocopaw.pipeline.ProcessingQueue] closes the burst.
 */
class DebugCapturePipeline(
    private val delegate: ImagePipeline,
    private val debugRoot: File,
    private val packageName: String,
    private val versionName: String,
    private val aspect: AspectRatio,
    private val exposurePlan: ExposurePlan,
    private val liveExposure: LiveExposure,
) : ImagePipeline {
    override val id: String get() = delegate.id
    override val framesPerCapture: Int get() = delegate.framesPerCapture

    override fun exposureFor(requested: ExposurePlan, live: LiveExposure, camera: CameraInfo): ExposurePlan =
        delegate.exposureFor(requested, live, camera)

    override fun openStream(exposure: ExposurePlan): BurstStream? = null

    override suspend fun process(burst: RawBurst, sink: OutputSink, progress: (Float) -> Unit): PipelineResult {
        var shotDir: File? = null
        var baseManifest: CaptureDebugManifest? = null
        withContext(Dispatchers.IO) {
            runCatching {
                val dir = DebugCapture.createShotDir(debugRoot, burst.metadata.captureTimeMillis, burst.metadata.cameraId)
                shotDir = dir
                baseManifest = DebugCapture.dumpRawAndSnapshot(
                    shotDir = dir,
                    burst = burst,
                    packageName = packageName,
                    versionName = versionName,
                    aspect = aspect,
                    exposurePlan = exposurePlan,
                    liveExposure = liveExposure,
                    pipelineId = delegate.id,
                )
            }.onFailure(DebugCapture::logFailed)
        }
        progress(0.15f)

        var result: PipelineResult? = null
        var failure: Throwable? = null
        try {
            result = delegate.process(burst, sink) { p ->
                progress(0.15f + 0.80f * p.coerceIn(0f, 1f))
            }
            return result
        } catch (t: Throwable) {
            failure = t
            throw t
        } finally {
            val dir = shotDir
            val initial = baseManifest
            if (dir != null && initial != null) {
                withContext(NonCancellable + Dispatchers.IO) {
                    runCatching {
                        val ref = burst.referenceIndex
                        val completed = initial.copy(
                            referenceFrameIndex = ref,
                            totalPipelineMs = result?.totalMillis,
                            stageTimings = result?.timings?.map { StageTimingJson(it.stage, it.millis) }.orEmpty(),
                            savedOutputs = result?.outputs?.map { SavedOutputJson(it.kind.name, it.displayName, it.uri.toString()) }.orEmpty(),
                            pipelineError = failure?.let { "${it::class.simpleName}: ${it.message}" },
                            frames = initial.frames.map { f -> f.copy(isReference = f.index == ref) },
                        )
                        DebugCapture.writeManifest(dir, completed)
                        DebugCapture.logSaved(dir, burst.size)
                    }.onFailure(DebugCapture::logFailed)
                }
            }
            if (failure == null) progress(1f)
        }
    }
}
