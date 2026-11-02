package com.izum1k.pocopaw.pipeline

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
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
) {
    private val released = AtomicBoolean(false)

    fun release() {
        if (released.compareAndSet(false, true)) NativeMemory.free(data)
    }

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
) {
    companion object {
        fun from(
            cameraId: String,
            facing: LensFacing,
            characteristics: CameraCharacteristics,
            cfa: CfaPattern,
            result: CaptureResult,
            orientationDegrees: Int,
            zoomRatio: Float,
            cropZoom: Float,
        ): RawMetadata {
            val pattern = characteristics.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
            val black = result.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
                ?: FloatArray(4) { i -> pattern?.getOffsetForIndex(i % 2, i / 2)?.toFloat() ?: 0f }
            val white = result.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL)
                ?: characteristics.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023
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
                captureTimeMillis = System.currentTimeMillis(),
            )
        }
    }
}

/** Frames of one shutter press. A single-frame capture is a burst of one. Closing it frees the frames. */
class RawBurst(
    val frames: List<RawFrame>,
    val metadata: RawMetadata,
    val referenceIndex: Int = 0,
) : Closeable {
    override fun close() = frames.forEach { it.release() }

    init {
        require(frames.isNotEmpty()) { "A burst needs at least one frame" }
        require(referenceIndex in frames.indices)
    }

    val reference: RawFrame get() = frames[referenceIndex]
    val size: Int get() = frames.size
}
