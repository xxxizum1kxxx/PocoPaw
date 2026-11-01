package com.izum1k.pocopaw.camera

import android.graphics.RectF
import android.media.Image
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** A region scored during a focus sweep, normalised to the frame in the sensor's orientation. */
data class FocusZone(val rect: RectF, val weight: Float)

/** One sweep step: the distance asked for, the one the result reported, and each zone's sharpness. */
class FocusSample(val step: Int, val target: Float, val dioptres: Float, val stationary: Boolean, val scores: FloatArray)

/** Where a sweep settled: [dioptres], how distinct its peak was (max / min score), steps and time taken. */
data class SweepResult(val dioptres: Float, val contrast: Float, val steps: Int, val millis: Long)

/**
 * Contrast focus measured by the app on the YUV analysis stream: the same scene swept the same way
 * gives the same distance, whatever the HAL's AF did before.
 */
object FocusSweeper {

    /** A peak whose max / min score is below this is a flat scene (a wall, darkness): no answer. */
    const val FLAT_CONTRAST = 1.15f

    /** A fine sweep changing less than this across its steps is noise only. */
    const val FINE_FLAT_CONTRAST = 1.02f

    /** The middle half of the frame in 3 x 3 cells, the centre weighted most. */
    fun centreZones(): List<FocusZone> = buildList {
        for (j in 0 until 3) for (i in 0 until 3) {
            val l = 0.25f + i / 6f
            val t = 0.25f + j / 6f
            val w = when {
                i == 1 && j == 1 -> 4f
                i == 1 || j == 1 -> 2f
                else -> 1f
            }
            add(FocusZone(RectF(l, t, l + 1 / 6f, t + 1 / 6f), w))
        }
    }.sortedByDescending { it.weight }

    /**
     * Far to near: 0.5 D steps out to 4 D (25 cm), where a scene's subjects mostly are, then
     * 1.5–2 D steps to the closest focus. On the analysis stream a peak is about 2 D wide.
     */
    fun coarseSteps(minFocusDioptres: Float): List<Float> {
        val out = ArrayList<Float>()
        var d = 0f
        while (d <= min(4f, minFocusDioptres) + 1e-3f) {
            out += d
            d += 0.5f
        }
        for (near in listOf(5f, 6.5f, 8f, 10f, 12f, 14f, 16f, 18f, 20f)) {
            if (near < minFocusDioptres - 0.5f) out += near
        }
        if (out.last() < minFocusDioptres - 1e-3f) out += minFocusDioptres
        return out
    }

    /** Steps between [lo] and [hi], about [step] apart, 5 to 7 of them. */
    fun fineSteps(lo: Float, hi: Float, step: Float, minFocusDioptres: Float): List<Float> {
        val a = lo.coerceIn(0f, minFocusDioptres)
        val b = hi.coerceIn(0f, minFocusDioptres)
        val n = ((b - a) / step).toInt().coerceIn(4, 6)
        return List(n + 1) { a + (b - a) * it / n }
    }

    /**
     * Each zone's Tenengrad (squared Sobel gradient) of a [1 2 1]-blurred luma, over the zone's
     * mean luma squared so exposure drift during a sweep does not count.
     */
    fun score(image: Image, zones: List<FocusZone>): FloatArray {
        val plane = image.planes[0]
        val buf = plane.buffer
        return FloatArray(zones.size) { zoneScore(buf, plane.rowStride, image.width, image.height, zones[it].rect) }
    }

    private fun zoneScore(buf: ByteBuffer, rowStride: Int, w: Int, h: Int, r: RectF): Float {
        val x0 = (r.left * w).toInt().coerceIn(2, w - 4)
        val x1 = (r.right * w).toInt().coerceIn(x0 + 2, w - 2)
        val y0 = (r.top * h).toInt().coerceIn(2, h - 4)
        val y1 = (r.bottom * h).toInt().coerceIn(y0 + 2, h - 2)
        val pw = x1 - x0 + 4
        val ph = y1 - y0 + 4
        val px = IntArray(pw * ph)
        val row = ByteArray(pw)
        for (j in 0 until ph) {
            buf.position((y0 - 2 + j) * rowStride + x0 - 2)
            buf.get(row, 0, pw)
            val o = j * pw
            for (i in 0 until pw) px[o + i] = row[i].toInt() and 0xFF
        }
        val t = IntArray(pw * ph)
        for (j in 0 until ph) {
            val o = j * pw
            for (i in 1 until pw - 1) t[o + i] = px[o + i - 1] + 2 * px[o + i] + px[o + i + 1]
        }
        val b = IntArray(pw * ph)
        for (j in 1 until ph - 1) {
            val o = j * pw
            for (i in 1 until pw - 1) b[o + i] = t[o + i - pw] + 2 * t[o + i] + t[o + i + pw]
        }
        var grad = 0.0
        var luma = 0.0
        var n = 0
        for (j in 2 until ph - 2) {
            val o = j * pw
            for (i in 2 until pw - 2) {
                val k = o + i
                val gx = (b[k - pw + 1] + 2 * b[k + 1] + b[k + pw + 1]) - (b[k - pw - 1] + 2 * b[k - 1] + b[k + pw - 1])
                val gy = (b[k + pw - 1] + 2 * b[k + pw] + b[k + pw + 1]) - (b[k - pw - 1] + 2 * b[k - pw] + b[k - pw + 1])
                grad += gx.toDouble() * gx + gy.toDouble() * gy
                luma += b[k]
                n++
            }
        }
        val mean = luma / n
        return if (mean < 1.0) 0f else (grad / n / (mean * mean)).toFloat()
    }

    /**
     * The zones' curves combined: each zone scaled to 0..1 over the sweep and weighted by its own
     * weight times how much it changed, so a flat wall's noise does not steer the pick.
     */
    fun combined(samples: List<FocusSample>, zones: List<FocusZone>): FloatArray {
        val out = FloatArray(samples.size)
        for (z in zones.indices) {
            val c = FloatArray(samples.size) { samples[it].scores[z] }
            val hi = c.max()
            val lo = c.min()
            if (lo <= 0f || hi <= lo) continue
            val w = zones[z].weight * (hi / lo - 1f).coerceIn(0f, 1f)
            for (k in c.indices) out[k] += w * (c[k] - lo) / (hi - lo)
        }
        return out
    }

    /** max / min of the weight-summed raw scores. */
    fun contrast(samples: List<FocusSample>, zones: List<FocusZone>): Float {
        val raw = FloatArray(samples.size) { k -> zones.indices.sumOf { (zones[it].weight * samples[k].scores[it]).toDouble() }.toFloat() }
        val lo = raw.min()
        return if (lo <= 0f) 0f else raw.max() / lo
    }

    /**
     * The sample to focus on. Of the combined curve's local maxima within 60% of the highest, the
     * one where the first (heaviest) zone is sharpest: with a near object and the wall behind both
     * in the frame, the centre decides.
     */
    fun pick(samples: List<FocusSample>, zones: List<FocusZone>): Int {
        val c = combined(samples, zones)
        val top = c.max()
        if (top <= 0f) return samples.indices.maxBy { samples[it].scores.sum() }
        val peaks = c.indices.filter { k ->
            c[k] >= 0.6f * top && (k == 0 || c[k] >= c[k - 1]) && (k == c.lastIndex || c[k] >= c[k + 1])
        }
        if (peaks.size <= 1 || zones.size == 1) return c.indices.maxBy { c[it] }
        return peaks.maxBy { samples[it].scores[0] }
    }

    /**
     * The vertex of the least-squares parabola through every sample's weight-summed score, in
     * dioptres: steadier than three points when the steps are finer than the noise. Null when the
     * fit is not a peak inside the swept range.
     */
    fun fitPeak(samples: List<FocusSample>, zones: List<FocusZone>): Float? {
        if (samples.size < 4) return null
        val xs = DoubleArray(samples.size) { samples[it].dioptres.toDouble() }
        val ys = DoubleArray(samples.size) { k -> zones.indices.sumOf { (zones[it].weight * samples[k].scores[it]).toDouble() } }
        val mx = xs.average()
        var s1 = 0.0; var s2 = 0.0; var s3 = 0.0; var s4 = 0.0
        var t0 = 0.0; var t1 = 0.0; var t2 = 0.0
        for (k in xs.indices) {
            val x = xs[k] - mx
            val y = ys[k]
            s1 += x; s2 += x * x; s3 += x * x * x; s4 += x * x * x * x
            t0 += y; t1 += x * y; t2 += x * x * y
        }
        val n = xs.size.toDouble()
        // normal equations for y = a x^2 + b x + c, by Cramer's rule
        val det = s4 * (s2 * n - s1 * s1) - s3 * (s3 * n - s1 * s2) + s2 * (s3 * s1 - s2 * s2)
        if (abs(det) < 1e-12) return null
        val a = (t2 * (s2 * n - s1 * s1) - s3 * (t1 * n - s1 * t0) + s2 * (t1 * s1 - s2 * t0)) / det
        val b = (s4 * (t1 * n - s1 * t0) - t2 * (s3 * n - s1 * s2) + s2 * (s3 * t0 - t1 * s2)) / det
        if (a >= 0.0) return null
        val v = -b / (2 * a) + mx
        return if (v < xs.min() || v > xs.max()) null else v.toFloat()
    }

    /** The vertex of the parabola through the pick and its neighbours, in dioptres. */
    fun interpolate(samples: List<FocusSample>, zones: List<FocusZone>, i: Int): Float {
        if (i <= 0 || i >= samples.lastIndex) return samples[i].dioptres
        val c = combined(samples, zones)
        val x0 = samples[i - 1].dioptres
        val x1 = samples[i].dioptres
        val x2 = samples[i + 1].dioptres
        val y0 = c[i - 1]
        val y1 = c[i]
        val y2 = c[i + 1]
        val denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if (abs(denom) < 1e-9f) return x1
        val a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        val b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
        if (a >= 0f) return x1
        return (-b / (2 * a)).coerceIn(min(x0, x2), max(x0, x2))
    }
}
