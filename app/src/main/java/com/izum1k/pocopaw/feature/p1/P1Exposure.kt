package com.izum1k.pocopaw.feature.p1

import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/** Per-frame exposures for a [P1Variant], computed as probe's `CaptureActivity.rawBurst()` does. */
object P1Exposure {
    private const val NIGHT_BLUR_PX = 1.5
    private const val NIGHT_OIS_CREDIT = 4.0
    private const val NIGHT_MIN_S = 0.033
    private const val NIGHT_TOTAL_S = 3.0

    /** Null until the preview AE has reported (dark frames need no metering). */
    fun plan(v: P1Variant, live: LiveExposure, cam: CameraInfo, gyroRms: Double): List<ManualExposure>? {
        if (v.mode == P1Mode.DARK) {
            val n = v.totalFrames
            return List(n) { i ->
                val iso = v.darkIsos[min(v.darkIsos.lastIndex, i * v.darkIsos.size / n)]
                ManualExposure(cam.isoRange?.let { iso.coerceIn(it.lower, it.upper) } ?: iso, v.darkExposureNs)
            }
        }
        val e0 = live.exposureNs ?: return null
        val iso0 = live.iso ?: return null
        if (e0 <= 0 || iso0 <= 0) return null
        val isoLo = cam.isoRange?.lower ?: iso0
        val isoHi = cam.isoRange?.upper ?: iso0
        // The HAL clamps an ISO past the sensor's range; the shutter then has to come from the clamped one.
        val fixedIso = v.iso?.coerceIn(isoLo, isoHi)
        val expLo = cam.exposureRangeNs?.lower ?: 0L
        val expHi = cam.exposureRangeNs?.upper ?: Long.MAX_VALUE
        var cap = min(v.maxExposureNs, expHi)
        var n = v.frames
        if (v.mode == P1Mode.NIGHT) {
            val night = nightExposureNs(v, cam, gyroRms)
            cap = min(cap, night)
            n = (NIGHT_TOTAL_S / (night / 1e9)).roundToInt().coerceIn(6, 30)
        }
        return List(n) { i ->
            val stops = v.ev + (v.bracket?.let { it[i % it.size] } ?: 0.0)
            val total = e0.toDouble() * iso0 * 2.0.pow(stops)
            if (fixedIso != null) {
                ManualExposure(fixedIso, (total / fixedIso).toLong().coerceIn(expLo, expHi))
            } else {
                var iso = isoLo
                var exp = (total / iso).toLong()
                if (exp > cap) {
                    exp = cap
                    iso = min(isoHi, ceil(total / exp).toInt())
                }
                ManualExposure(iso, max(expLo, exp))
            }
        }
    }

    private fun nightExposureNs(v: P1Variant, cam: CameraInfo, gyroRms: Double): Long {
        val sensor = cam.sensorSizeMm ?: return v.maxExposureNs
        val focalPx = cam.focalLengthMm / (sensor.width / cam.rawSize.width)
        val budget = NIGHT_BLUR_PX * if (cam.hasOis) NIGHT_OIS_CREDIT else 1.0
        val t = budget / (max(gyroRms, 0.005) * focalPx)
        return (max(NIGHT_MIN_S, min(v.maxExposureNs / 1e9, t)) * 1e9).toLong()
    }
}
