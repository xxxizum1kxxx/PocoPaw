package com.izum1k.pocopaw.pipeline

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.Face
import android.hardware.camera2.params.LensShadingMap
import android.hardware.camera2.params.RggbChannelVector
import android.util.Rational
import com.izum1k.pocopaw.camera.CfaPattern
import com.izum1k.pocopaw.camera.LensFacing
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * One RAW_SENSOR frame copied out of the camera's buffer: 16-bit samples, native byte order,
 * tightly packed (row stride = width × 2), so the camera's Image can be released at once.
 * [data] lives in [NativeMemory]; [release] frees it and no one may touch it afterwards.
 */
class RawFrame(
    val data: ByteBuffer,
    val width: Int,
    val height: Int,
    val timestampNs: Long,
    val result: CaptureResult,
    private val onRelease: (ByteBuffer) -> Unit = { NativeMemory.free(it) },
) : Closeable {
    private val released = AtomicBoolean(false)

    fun release() {
        if (released.compareAndSet(false, true)) onRelease(data)
    }

    override fun close() = release()

    val iso: Int? get() = result.get(CaptureResult.SENSOR_SENSITIVITY)
    val exposureNs: Long? get() = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)

    fun samples(): ShortBuffer = data.duplicate().order(ByteOrder.nativeOrder()).asShortBuffer()

    fun sample(x: Int, y: Int): Int = data.getShort((y * width + x) * 2).toInt() and 0xFFFF
}

/** Everything about a burst the processing needs and that is the same for every frame. */
class RawMetadata(
    val cameraId: String,
    val facing: LensFacing,
    val characteristics: CameraCharacteristics,
    val cfa: CfaPattern,
    /** Per CFA position, row-major over the 2×2 tile: dynamic black level when the HAL reports one. */
    val blackLevel: FloatArray,
    val whiteLevel: Int,
    val wbGains: RggbChannelVector?,
    val colorTransform: ColorSpaceTransform?,
    val neutralPoint: Array<Rational>?,
    val lensShading: LensShadingMap?,
    val hasNoiseProfile: Boolean,
    /** Clockwise rotation that makes the frame upright, as written to the DNG orientation tag. */
    val orientationDegrees: Int,
    /** The viewfinder's zoom; RAW always keeps the full sensor, so a later stage may crop to it. */
    val zoomRatio: Float,
    /** Digital zoom the processed image is cropped to (the 2x lens step); 1 keeps the full sensor. */
    val cropZoom: Float,
    val captureTimeMillis: Long,
    /** Normalized sensor-orientation face boxes [x0, y0, x1, y1, ...] from STATISTICS_FACES. */
    val faceBoxes: FloatArray = FloatArray(0),
) {
    fun withResult(result: CaptureResult): RawMetadata =
        from(cameraId, facing, characteristics, cfa, result, orientationDegrees, zoomRatio, cropZoom, captureTimeMillis)

    internal data class FaceBoxInput(val left: Int, val top: Int, val right: Int, val bottom: Int, val score: Int)

    companion object {
        internal const val MIN_FACE_SCORE = 50

        internal fun normalizeFaceBoxes(
            faces: List<FaceBoxInput>,
            activeWidth: Int,
            activeHeight: Int,
            minScore: Int = MIN_FACE_SCORE,
            activeLeft: Int = 0,
            activeTop: Int = 0,
        ): FloatArray {
            if (faces.isEmpty() || activeWidth <= 0 || activeHeight <= 0) return FloatArray(0)
            val w = activeWidth.toFloat()
            val h = activeHeight.toFloat()
            val l = activeLeft.toFloat()
            val t = activeTop.toFloat()
            val out = ArrayList<Float>(faces.size * 4)
            for (f in faces) {
                if (f.score < minScore) continue
                val x0 = ((minOf(f.left, f.right) - l) / w).coerceIn(0f, 1f)
                val y0 = ((minOf(f.top, f.bottom) - t) / h).coerceIn(0f, 1f)
                val x1 = ((maxOf(f.left, f.right) - l) / w).coerceIn(0f, 1f)
                val y1 = ((maxOf(f.top, f.bottom) - t) / h).coerceIn(0f, 1f)
                if (x1 - x0 < 1e-4f || y1 - y0 < 1e-4f) continue
                out.add(x0)
                out.add(y0)
                out.add(x1)
                out.add(y1)
            }
            return out.toFloatArray()
        }

        internal fun normalizeFaceBoxes(
            faces: Array<Face>?,
            activeWidth: Int,
            activeHeight: Int,
            minScore: Int = MIN_FACE_SCORE,
            activeLeft: Int = 0,
            activeTop: Int = 0,
        ): FloatArray {
            if (faces == null || faces.isEmpty() || activeWidth <= 0 || activeHeight <= 0) return FloatArray(0)
            val inputs = faces.mapNotNull { face ->
                val b = face.bounds ?: return@mapNotNull null
                FaceBoxInput(b.left, b.top, b.right, b.bottom, face.score)
            }
            return normalizeFaceBoxes(inputs, activeWidth, activeHeight, minScore, activeLeft, activeTop)
        }

        fun from(
            cameraId: String,
            facing: LensFacing,
            characteristics: CameraCharacteristics,
            cfa: CfaPattern,
            result: CaptureResult,
            orientationDegrees: Int,
            zoomRatio: Float,
            cropZoom: Float,
            captureTimeMillis: Long = System.currentTimeMillis(),
        ): RawMetadata {
            val pattern = characteristics.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
            val black = result.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
                ?: FloatArray(4) { i -> pattern?.getOffsetForIndex(i % 2, i / 2)?.toFloat() ?: 0f }
            val white = result.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL)
                ?: characteristics.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023
            val active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            val faceBoxes = normalizeFaceBoxes(
                result.get(CaptureResult.STATISTICS_FACES),
                active?.width() ?: 0,
                active?.height() ?: 0,
                activeLeft = active?.left ?: 0,
                activeTop = active?.top ?: 0,
            )
            return RawMetadata(
                cameraId = cameraId,
                facing = facing,
                characteristics = characteristics,
                cfa = cfa,
                blackLevel = black,
                whiteLevel = white,
                wbGains = result.get(CaptureResult.COLOR_CORRECTION_GAINS),
                colorTransform = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM),
                neutralPoint = result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT),
                lensShading = result.get(CaptureResult.STATISTICS_LENS_SHADING_CORRECTION_MAP),
                hasNoiseProfile = result.get(CaptureResult.SENSOR_NOISE_PROFILE) != null,
                orientationDegrees = orientationDegrees,
                zoomRatio = zoomRatio,
                cropZoom = cropZoom,
                captureTimeMillis = captureTimeMillis,
                faceBoxes = faceBoxes,
            )
        }
    }
}

/**
 * Frames of one shutter press. A single-frame capture is a burst of one. Closing it frees the frames.
 * With [stream] the frames were merged while captured and all but the reference are already released.
 */
class RawBurst(
    val frames: List<RawFrame>,
    metadata: RawMetadata,
    referenceIndex: Int = 0,
    val stream: BurstStream? = null,
) : Closeable {
    var metadata: RawMetadata = metadata
        private set

    override fun close() {
        stream?.close()
        frames.forEach { it.release() }
    }

    /** Set by [ReferenceStage] before the other stages read [reference] and [metadata]. */
    var referenceIndex: Int = referenceIndex
        set(value) {
            require(value in frames.indices)
            field = value
            metadata = metadata.withResult(frames[value].result)
        }

    init {
        require(frames.isNotEmpty()) { "A burst needs at least one frame" }
        require(referenceIndex in frames.indices)
        this.metadata = metadata.withResult(frames[referenceIndex].result)
    }

    val reference: RawFrame get() = frames[referenceIndex]
    val size: Int get() = frames.size
}
