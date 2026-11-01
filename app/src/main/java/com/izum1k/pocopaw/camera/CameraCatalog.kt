package com.izum1k.pocopaw.camera

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureResult
import android.util.Range
import android.util.Rational
import android.util.Size
import android.util.SizeF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.roundToInt

enum class LensFacing { BACK, FRONT, EXTERNAL }

enum class CfaPattern { RGGB, GRBG, GBRG, BGGR, UNKNOWN }

/** One physical camera that can deliver RAW, with what the UI and the pipeline need to know. */
data class CameraInfo(
    val id: String,
    val facing: LensFacing,
    val characteristics: CameraCharacteristics,
    val rawSize: Size,
    val previewSize: Size,
    val activeArray: Rect,
    val cfa: CfaPattern,
    val whiteLevel: Int,
    val blackLevel: Int,
    val focalLengthMm: Float,
    val aperture: Float?,
    val equivalentFocalMm: Float,
    val sensorSizeMm: SizeF?,
    val sensorOrientation: Int,
    val isoRange: Range<Int>?,
    val exposureRangeNs: Range<Long>?,
    val evRange: Range<Int>,
    val evStep: Rational,
    val zoomRange: Range<Float>,
    val hasFlash: Boolean,
    val hasOis: Boolean,
    val hasAutoFocus: Boolean,
    val hardwareLevel: Int,
    val hasNoiseProfile: Boolean,
    /** Field of view relative to the main back camera: 1 for the main camera, < 1 for ultra-wide. */
    val relativeZoom: Float,
) {
    val rawMegapixels: Float get() = rawSize.width * rawSize.height / 1_000_000f
}

/** Physical RAW cameras that are safe to open, main back camera first. */
class CameraCatalog(private val manager: CameraManager) {

    val cameras: List<CameraInfo> by lazy { load() }

    val back: List<CameraInfo> get() = cameras.filter { it.facing == LensFacing.BACK }.sortedBy { it.relativeZoom }
    val front: List<CameraInfo> get() = cameras.filter { it.facing == LensFacing.FRONT }

    fun byId(id: String?): CameraInfo? = cameras.firstOrNull { it.id == id }

    fun default(): CameraInfo? = back.firstOrNull { it.relativeZoom == 1f } ?: cameras.firstOrNull()

    private fun load(): List<CameraInfo> {
        val listed = runCatching { manager.cameraIdList.toList() }.getOrDefault(emptyList())
        val ids = (listed + PROBED_IDS).distinct().filterNot { it in DENYLIST }
        val raw = ids.mapNotNull { id ->
            val c = runCatching { manager.getCameraCharacteristics(id) }.getOrNull() ?: return@mapNotNull null
            describe(id, c)
        }
        val main = raw.filter { it.facing == LensFacing.BACK }.minByOrNull { it.id.toIntOrNull() ?: Int.MAX_VALUE }
        val mainEq = main?.equivalentFocalMm ?: 0f
        return raw.map { info ->
            val zoom = if (info.facing == LensFacing.BACK && mainEq > 0f) roundZoom(info.equivalentFocalMm / mainEq) else 1f
            info.copy(relativeZoom = zoom)
        }.sortedWith(compareBy({ it.facing.ordinal }, { it.relativeZoom }))
    }

    private fun describe(id: String, c: CameraCharacteristics): CameraInfo? {
        val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: return null
        if (CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW !in caps) return null
        if (CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA in caps) return null
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val rawSize = map.getOutputSizes(ImageFormat.RAW_SENSOR)?.maxByOrNull { it.width.toLong() * it.height } ?: return null
        val previewSize = choosePreviewSize(map.getOutputSizes(SurfaceTexture::class.java).orEmpty().toList(), rawSize)
        val facing = when (c.get(CameraCharacteristics.LENS_FACING)) {
            CameraMetadata.LENS_FACING_FRONT -> LensFacing.FRONT
            CameraMetadata.LENS_FACING_BACK -> LensFacing.BACK
            else -> LensFacing.EXTERNAL
        }
        val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
        val sensor = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val pixelArray = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        val active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: Rect(0, 0, rawSize.width, rawSize.height)
        val equivalent = if (sensor != null && pixelArray != null && focal > 0f) {
            val activeDiagMm = hypot(
                sensor.width * active.width() / pixelArray.width.toFloat(),
                sensor.height * active.height() / pixelArray.height.toFloat(),
            )
            focal * FULL_FRAME_DIAGONAL_MM / activeDiagMm
        } else {
            focal
        }
        val cfa = when (c.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)) {
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> CfaPattern.RGGB
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> CfaPattern.GRBG
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> CfaPattern.GBRG
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> CfaPattern.BGGR
            else -> CfaPattern.UNKNOWN
        }
        val ois = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
            ?.any { it == CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON } == true
        val minFocus = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val zoomRange = c.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
            ?: Range(1f, c.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f)
        return CameraInfo(
            id = id,
            facing = facing,
            characteristics = c,
            rawSize = rawSize,
            previewSize = previewSize,
            activeArray = active,
            cfa = cfa,
            whiteLevel = c.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 1023,
            blackLevel = c.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.getOffsetForIndex(0, 0) ?: 0,
            focalLengthMm = focal,
            aperture = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.firstOrNull(),
            equivalentFocalMm = equivalent,
            sensorSizeMm = sensor,
            sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90,
            isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE),
            exposureRangeNs = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE),
            evRange = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE) ?: Range(0, 0),
            evStep = c.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_STEP) ?: Rational(1, 3),
            zoomRange = zoomRange,
            hasFlash = c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true,
            hasOis = ois,
            hasAutoFocus = minFocus > 0f,
            hardwareLevel = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: -1,
            hasNoiseProfile = CaptureResult.SENSOR_NOISE_PROFILE in c.availableCaptureResultKeys,
            relativeZoom = 1f,
        )
    }

    private fun choosePreviewSize(sizes: List<Size>, raw: Size): Size {
        val aspect = raw.width.toFloat() / raw.height
        return sizes
            .filter { abs(it.width.toFloat() / it.height - aspect) < 0.02f && it.width <= MAX_PREVIEW_WIDTH }
            .maxByOrNull { it.width.toLong() * it.height }
            ?: sizes.filter { it.width <= MAX_PREVIEW_WIDTH }.maxByOrNull { it.width.toLong() * it.height }
            ?: Size(1440, 1080)
    }

    private fun roundZoom(z: Float): Float = if (abs(z - 1f) < 0.08f) 1f else (z * 10f).roundToInt() / 10f

    companion object {
        /** 4–6 are logical/auxiliary cameras that hang or crash marble's HAL; 3 is the 2 MP macro. */
        val DENYLIST = setOf("3", "4", "5", "6")

        /** Auxiliary physical cameras LineageOS exposes without listing them in cameraIdList. */
        private val PROBED_IDS = listOf("0", "1", "2")

        private const val FULL_FRAME_DIAGONAL_MM = 43.27f
        private const val MAX_PREVIEW_WIDTH = 1920
    }
}
