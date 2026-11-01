package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToLong

/**
 * HDR+ style exposure for a burst: keep the preview AE's exposure × ISO product, under-expose by
 * [UNDEREXPOSURE_STOPS] to protect highlights (the merge recovers the shadows), then spend it on
 * the lowest ISO with a shutter no longer than the handheld limit, raising ISO only past it.
 * Auto exposure on a capture request would hold the shutter at 10 ms (50 Hz anti-banding on
 * marble) and pay for it in ISO.
 */
object BurstExposure {
    const val UNDEREXPOSURE_STOPS = -1.0
    const val MAX_EXPOSURE_OIS_NS = 66_000_000L
    const val MAX_EXPOSURE_NS = 33_000_000L

    /** Bracket: [BRACKET_BASE_STOPS] for the short frames, every third frame [BRACKET_LONG_STOPS] longer. */
    const val BRACKET_BASE_STOPS = -2.0
    const val BRACKET_LONG_STOPS = 2.0

    /** Night: metered exposure, per-frame shutter limited by the expected blur instead of a fixed cap. */
    const val NIGHT_BLUR_PX = 1.5
    const val NIGHT_OIS_CREDIT = 4.0
    const val NIGHT_MAX_EXPOSURE_NS = 333_000_000L
    const val NIGHT_MIN_EXPOSURE_NS = 33_000_000L

    fun plan(live: LiveExposure, camera: CameraInfo, evIndex: Int = 0): ExposurePlan.Manual? =
        meter(live, camera, UNDEREXPOSURE_STOPS + evIndex * camera.evStep.toDouble(), handheldLimit(camera))?.let { ExposurePlan.Manual(it) }

    /**
     * Two short frames then one [BRACKET_LONG_STOPS] longer, cycled. The native merge picks the
     * shortest frame as reference (it keeps the highlights) and scales the others to it.
     */
    fun bracket(live: LiveExposure, camera: CameraInfo, evIndex: Int = 0): ExposurePlan.PerFrame? {
        val stops = BRACKET_BASE_STOPS + evIndex * camera.evStep.toDouble()
        val short = meter(live, camera, stops, handheldLimit(camera)) ?: return null
        val long = meter(live, camera, stops + BRACKET_LONG_STOPS, handheldLimit(camera)) ?: return null
        return ExposurePlan.PerFrame(listOf(short, short, long))
    }

    /**
     * Motion metering (probe night mode): the longest shutter whose blur, gyro RMS rate × focal length
     * in pixels, stays within [NIGHT_BLUR_PX] (× [NIGHT_OIS_CREDIT] with OIS), at the metered exposure.
     */
    fun night(live: LiveExposure, camera: CameraInfo, evIndex: Int = 0): ExposurePlan.Manual? {
        val sensor = camera.sensorSizeMm ?: return plan(live, camera, evIndex)
        val focalPx = camera.focalLengthMm / (sensor.width / camera.rawSize.width)
        val budget = NIGHT_BLUR_PX * if (camera.hasOis) NIGHT_OIS_CREDIT else 1.0
        val rate = max(live.motionRadPerSec?.toDouble() ?: 0.02, 0.005)
        val limitNs = (budget / (rate * focalPx) * 1e9).roundToLong()
            .coerceIn(NIGHT_MIN_EXPOSURE_NS, NIGHT_MAX_EXPOSURE_NS)
            .coerceAtMost(camera.exposureRangeNs?.upper ?: Long.MAX_VALUE)
        return meter(live, camera, evIndex * camera.evStep.toDouble(), limitNs)?.let { ExposurePlan.Manual(it) }
    }

    private fun handheldLimit(camera: CameraInfo): Long =
        (if (camera.hasOis) MAX_EXPOSURE_OIS_NS else MAX_EXPOSURE_NS).coerceAtMost(camera.exposureRangeNs?.upper ?: Long.MAX_VALUE)

    private fun meter(live: LiveExposure, camera: CameraInfo, stops: Double, limit: Long): ManualExposure? {
        val iso0 = live.iso ?: return null
        val exp0 = live.exposureNs ?: return null
        if (iso0 <= 0 || exp0 <= 0) return null
        val total = exp0.toDouble() * iso0 * 2.0.pow(stops)
        val isoRange = camera.isoRange
        val expRange = camera.exposureRangeNs
        val minIso = isoRange?.lower ?: iso0
        val maxIso = isoRange?.upper ?: iso0
        var iso = minIso
        var exp = (total / iso).roundToLong()
        if (exp > limit) {
            exp = limit
            iso = ceil(total / exp).toInt().coerceIn(minIso, maxIso)
        }
        exp = exp.coerceIn(expRange?.lower ?: exp, limit)
        return ManualExposure(iso, exp)
    }
}
