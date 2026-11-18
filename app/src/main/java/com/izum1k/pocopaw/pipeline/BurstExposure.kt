package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt
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
    const val HIGHLIGHT_CLIP_SAFE = 0.02f
    const val HIGHLIGHT_CLIP_FULL = 0.18f
    const val HIGHLIGHT_GUARD_MAX_STOPS = 1.0
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

    /** Boost frame count to 12–16 when digitally zooming (`cropZoom > 1.01f`) unless flash fires. */
    fun framesForZoom(baseFrames: Int, cropZoom: Float, flash: Boolean = false): Int {
        if (flash) return 1
        if (!cropZoom.isFinite() || cropZoom <= 1.01f) return baseFrames.coerceAtLeast(1)
        val z = cropZoom.coerceIn(1f, 2f)
        val boosted = 12 + ((z - 1f) * 4f).roundToInt()
        return maxOf(baseFrames, boosted).coerceIn(12, 16)
    }

    /** Clamp [desiredFrames] safely against [budget.maxAffordable] without throwing (returns `0` when exhausted). */
    fun framesWithinBudget(desiredFrames: Int, budget: com.izum1k.pocopaw.camera.RawRequestBudget): Int {
        val desired = desiredFrames.coerceAtLeast(1)
        val affordable = budget.maxAffordable()
        return if (affordable <= 0) 0 else desired.coerceAtMost(affordable)
    }

    fun plan(live: LiveExposure, camera: CameraInfo, evIndex: Int = 0): ExposurePlan.Manual? =
        planExposure(
            live = live,
            hasOis = camera.hasOis,
            evIndex = evIndex,
            evStep = camera.evStep.toDouble(),
            minIso = camera.isoRange?.lower,
            maxIso = camera.isoRange?.upper,
            minExp = camera.exposureRangeNs?.lower,
            maxExp = camera.exposureRangeNs?.upper,
        )?.let { ExposurePlan.Manual(it) }

    internal fun highlightGuardStops(clippedFraction: Float): Double {
        if (!clippedFraction.isFinite() || clippedFraction <= HIGHLIGHT_CLIP_SAFE) return 0.0
        val t = ((clippedFraction - HIGHLIGHT_CLIP_SAFE) / (HIGHLIGHT_CLIP_FULL - HIGHLIGHT_CLIP_SAFE))
            .toDouble()
            .coerceIn(0.0, 1.0)
        return -HIGHLIGHT_GUARD_MAX_STOPS * t * t * (3.0 - 2.0 * t)
    }

    internal fun planExposure(
        live: LiveExposure,
        hasOis: Boolean,
        evIndex: Int = 0,
        evStep: Double = 1.0 / 3.0,
        minIso: Int? = null,
        maxIso: Int? = null,
        minExp: Long? = null,
        maxExp: Long? = null,
    ): ManualExposure? {
        val iso0 = live.iso ?: return null
        val exp0 = live.exposureNs ?: return null
        val guard = if (evIndex == 0) highlightGuardStops(live.clippedFraction) else 0.0
        val limit = (if (hasOis) MAX_EXPOSURE_OIS_NS else MAX_EXPOSURE_NS).coerceAtMost(maxExp ?: Long.MAX_VALUE)
        return meterExposure(
            iso0 = iso0,
            exp0 = exp0,
            stops = UNDEREXPOSURE_STOPS + evIndex * evStep + guard,
            limit = limit,
            minIso = minIso ?: iso0,
            maxIso = maxIso ?: iso0,
            minExp = minExp,
        )
    }

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
        return meterExposure(
            iso0 = iso0,
            exp0 = exp0,
            stops = stops,
            limit = limit,
            minIso = camera.isoRange?.lower ?: iso0,
            maxIso = camera.isoRange?.upper ?: iso0,
            minExp = camera.exposureRangeNs?.lower,
        )
    }

    internal fun meterExposure(
        iso0: Int,
        exp0: Long,
        stops: Double,
        limit: Long,
        minIso: Int = iso0,
        maxIso: Int = iso0,
        minExp: Long? = null,
    ): ManualExposure? {
        if (iso0 <= 0 || exp0 <= 0 || limit <= 0) return null
        val loIso = minIso.coerceAtLeast(1)
        val hiIso = maxIso.coerceAtLeast(loIso)
        val total = exp0.toDouble() * iso0 * 2.0.pow(stops)
        var iso = loIso
        var exp = (total / iso).roundToLong()
        if (exp > limit) {
            exp = limit
            iso = ceil(total / exp).toInt().coerceIn(loIso, hiIso)
        }
        val lowerExp = minExp?.coerceAtMost(limit) ?: exp
        exp = exp.coerceIn(lowerExp, limit)
        return ManualExposure(iso, exp)
    }
}
