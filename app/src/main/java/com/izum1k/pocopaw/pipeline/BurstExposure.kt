package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import kotlin.math.ceil
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

    fun plan(live: LiveExposure, camera: CameraInfo, evIndex: Int = 0): ExposurePlan.Manual? {
        val iso0 = live.iso ?: return null
        val exp0 = live.exposureNs ?: return null
        if (iso0 <= 0 || exp0 <= 0) return null
        val stops = UNDEREXPOSURE_STOPS + evIndex * camera.evStep.toDouble()
        val total = exp0.toDouble() * iso0 * 2.0.pow(stops)
        val isoRange = camera.isoRange
        val expRange = camera.exposureRangeNs
        val minIso = isoRange?.lower ?: iso0
        val maxIso = isoRange?.upper ?: iso0
        val limit = (if (camera.hasOis) MAX_EXPOSURE_OIS_NS else MAX_EXPOSURE_NS).coerceAtMost(expRange?.upper ?: Long.MAX_VALUE)
        var iso = minIso
        var exp = (total / iso).roundToLong()
        if (exp > limit) {
            exp = limit
            iso = ceil(total / exp).toInt().coerceIn(minIso, maxIso)
        }
        exp = exp.coerceIn(expRange?.lower ?: exp, limit)
        return ExposurePlan.Manual(ManualExposure(iso, exp))
    }
}
