package com.izum1k.pocopaw.camera

import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.RggbChannelVector
import android.hardware.camera2.params.TonemapCurve
import android.media.Image
import com.izum1k.pocopaw.pipeline.RawFrame
import com.izum1k.pocopaw.pipeline.RawMetadata
import java.nio.ByteBuffer
import java.util.Locale
import kotlin.math.acos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/** Y, U and V bytes of [SceneMeter.COLS] x [SceneMeter.ROWS] cells, [SceneMeter.PTS] points each, copied out of an analysis frame. */
class MeterFrame(val y: ByteArray, val u: ByteArray, val v: ByteArray, val sampleNanos: Long)

/**
 * What the app measured on one analysis frame. [camRgb] is linear camera RGB before white balance
 * per cell (3 floats each, row-major), [valid] the cells with no clipped and no near-black point.
 */
class MeterStats(
    val timestampNs: Long,
    /** log2 of linear luma over ISO x seconds: the scene's brightness, whatever AE did. */
    val sceneEv: Double,
    /** Mean luma (0..1, as encoded) of [SceneMeter.GRID_COLS] x [SceneMeter.GRID_ROWS] cells, for a change of framing. */
    val grid: FloatArray,
    val camRgb: FloatArray,
    val valid: BooleanArray,
    /** The gains the frame was white balanced with (red, green, blue). */
    val gains: FloatArray,
    val tonemapFromResult: Boolean,
    /** Copying the points on the camera thread, and converting them off it. */
    val sampleMillis: Double,
    val millis: Double,
) {
    /** Grey world on [camRgb]: the illuminant's camera RGB, unit length, or null with too few valid cells. */
    fun greyWorld(): FloatArray? = SceneMeter.meanRgb(camRgb, valid, 0, SceneMeter.COLS, 0, SceneMeter.ROWS)

    /** The illuminant the frame's gains assume, in camera RGB, unit length. */
    fun gainsIlluminant(): FloatArray = SceneMeter.unit(floatArrayOf(1 / gains[0], 1 / gains[1], 1 / gains[2]))
}

/**
 * Brightness, framing and colour statistics of the YUV analysis stream. The stream has been
 * through the ISP (white balance gains, colour matrix, tone curve), so colour is taken back to
 * camera RGB with the frame's own gains, matrix and curve.
 */
object SceneMeter {

    const val COLS = 64
    const val ROWS = 48
    /** Points sampled per cell, on a 2 x 2 pattern. */
    const val PTS = 4
    const val GRID_COLS = 8
    const val GRID_ROWS = 6
    /** A point at or above this (0..1, encoded) in any channel is clipped; a cell whose brightest channel is below [DARK] is noise. */
    private const val CLIP = 0.90f
    private const val DARK = 0.002f
    private const val MIN_VALID_CELLS = 64

    /** Copies the points out of [image] so it can be released at once. */
    fun sample(image: Image): MeterFrame {
        val t0 = System.nanoTime()
        val w = image.width
        val h = image.height
        val yp = image.planes[0]
        val up = image.planes[1]
        val vp = image.planes[2]
        val n = COLS * ROWS * PTS
        val ys = ByteArray(n)
        val us = ByteArray(n)
        val vs = ByteArray(n)
        val yRow = ByteArray(yp.rowStride)
        val uRow = ByteArray(up.rowStride)
        val vRow = ByteArray(vp.rowStride)
        // whole rows in bulk: a ByteBuffer read per point costs about 10 ms a frame
        for (cy in 0 until ROWS) for (py in 0 until 2) {
            val y = ((cy + 0.25f + 0.5f * py) * h / ROWS).toInt().coerceIn(0, h - 1)
            readRow(yp.buffer, y * yp.rowStride, yRow)
            readRow(up.buffer, (y / 2) * up.rowStride, uRow)
            readRow(vp.buffer, (y / 2) * vp.rowStride, vRow)
            for (cx in 0 until COLS) for (px in 0 until 2) {
                val x = ((cx + 0.25f + 0.5f * px) * w / COLS).toInt().coerceIn(0, w - 1)
                val k = (cy * COLS + cx) * PTS + py * 2 + px
                ys[k] = yRow[x * yp.pixelStride]
                us[k] = uRow[(x / 2) * up.pixelStride]
                vs[k] = vRow[(x / 2) * vp.pixelStride]
            }
        }
        return MeterFrame(ys, us, vs, System.nanoTime() - t0)
    }

    private fun readRow(buf: ByteBuffer, offset: Int, dst: ByteArray) {
        val b = buf.duplicate()
        b.position(offset)
        b.get(dst, 0, min(dst.size, b.remaining()))
    }

    fun measure(f: MeterFrame, r: CaptureResult, timestampNs: Long): MeterStats? {
        val t0 = System.nanoTime()
        val g = r.get(CaptureResult.COLOR_CORRECTION_GAINS) ?: return null
        val t = r.get(CaptureResult.COLOR_CORRECTION_TRANSFORM) ?: return null
        val iso = r.get(CaptureResult.SENSOR_SENSITIVITY) ?: return null
        val exp = r.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: return null
        val boost = (r.get(CaptureResult.CONTROL_POST_RAW_SENSITIVITY_BOOST) ?: 100) / 100.0
        val curve = r.get(CaptureResult.TONEMAP_CURVE)
        val lut = if (curve != null) Array(3) { inverseLut(curve, it) } else Array(3) { SRGB_INVERSE }
        // COLOR_CORRECTION_TRANSFORM maps white-balanced camera RGB to linear sRGB
        val m = Array(3) { row -> DoubleArray(3) { col -> t.getElement(col, row).toDouble() } }
        val inv = invert(m) ?: return null
        val gains = floatArrayOf(g.red, (g.greenEven + g.greenOdd) / 2, g.blue)

        val cells = COLS * ROWS
        val camRgb = FloatArray(cells * 3)
        val valid = BooleanArray(cells)
        val grid = FloatArray(GRID_COLS * GRID_ROWS)
        var lumaSum = 0.0
        val lin = FloatArray(3)
        val rgb = FloatArray(3)
        for (c in 0 until cells) {
            var clipped = false
            var r0 = 0f
            var g0 = 0f
            var b0 = 0f
            var ySum = 0f
            for (p in 0 until PTS) {
                val k = c * PTS + p
                val yv = (f.y[k].toInt() and 0xFF).toFloat()
                val cb = (f.u[k].toInt() and 0xFF) - 128f
                val cr = (f.v[k].toInt() and 0xFF) - 128f
                ySum += yv
                rgb[0] = yv + 1.402f * cr
                rgb[1] = yv - 0.344136f * cb - 0.714136f * cr
                rgb[2] = yv + 1.772f * cb
                for (ch in 0 until 3) {
                    val e = (rgb[ch] / 255f).coerceIn(0f, 1f)
                    if (e >= CLIP) clipped = true
                    lin[ch] = lut[ch][(e * 255f + 0.5f).toInt()]
                }
                r0 += lin[0]
                g0 += lin[1]
                b0 += lin[2]
            }
            r0 /= PTS
            g0 /= PTS
            b0 /= PTS
            lumaSum += 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0
            val gi = (c / COLS) * GRID_ROWS / ROWS * GRID_COLS + (c % COLS) * GRID_COLS / COLS
            grid[gi] += ySum / PTS / 255f
            for (row in 0 until 3) {
                val bal = inv[row][0] * r0 + inv[row][1] * g0 + inv[row][2] * b0
                camRgb[c * 3 + row] = (bal / gains[row]).toFloat()
            }
            valid[c] = !clipped && max(r0, max(g0, b0)) > DARK
        }
        val perGridCell = cells / (GRID_COLS * GRID_ROWS)
        for (i in grid.indices) grid[i] /= perGridCell
        val luma = max(lumaSum / cells, 1e-6)
        val seconds = exp / 1e9
        val ev = ln(luma / (iso / 100.0 * boost * seconds)) / ln(2.0)
        return MeterStats(timestampNs, ev, grid, camRgb, valid, gains, curve != null, f.sampleNanos / 1e6, (System.nanoTime() - t0) / 1e6)
    }

    /** Pearson correlation of two grids: 1 for the same framing under any exposure. */
    fun correlation(a: FloatArray, b: FloatArray): Double {
        val n = a.size
        val ma = a.average()
        val mb = b.average()
        var sab = 0.0
        var saa = 0.0
        var sbb = 0.0
        for (i in 0 until n) {
            val da = a[i] - ma
            val db = b[i] - mb
            sab += da * db
            saa += da * da
            sbb += db * db
        }
        return if (saa < 1e-9 || sbb < 1e-9) 1.0 else sab / sqrt(saa * sbb)
    }

    /** Mean camera RGB of the valid cells in columns [x0, x1) and rows [y0, y1), unit length. */
    fun meanRgb(camRgb: FloatArray, valid: BooleanArray, x0: Int, x1: Int, y0: Int, y1: Int): FloatArray? {
        var r = 0.0
        var g = 0.0
        var b = 0.0
        var n = 0
        for (y in y0 until y1) for (x in x0 until x1) {
            val c = y * COLS + x
            if (!valid[c]) continue
            r += camRgb[c * 3]
            g += camRgb[c * 3 + 1]
            b += camRgb[c * 3 + 2]
            n++
        }
        val need = if (x1 - x0 == COLS && y1 - y0 == ROWS) MIN_VALID_CELLS else 1
        if (n < need || r <= 0 || g <= 0 || b <= 0) return null
        return unit(floatArrayOf(r.toFloat(), g.toFloat(), b.toFloat()))
    }

    fun unit(v: FloatArray): FloatArray {
        val n = sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
        return FloatArray(3) { v[it] / n }
    }

    fun angleDegrees(a: FloatArray, b: FloatArray): Double {
        val c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).toDouble() /
            sqrt(a.sumOf { (it * it).toDouble() } * b.sumOf { (it * it).toDouble() })
        return Math.toDegrees(acos(c.coerceIn(-1.0, 1.0)))
    }

    /** Gains (red, green, green, blue) that make [illuminant] neutral. */
    fun gainsFor(illuminant: FloatArray): RggbChannelVector {
        val g = illuminant[1]
        return RggbChannelVector(g / illuminant[0], 1f, 1f, g / illuminant[2])
    }

    /** Per-[GRID_COLS] x [GRID_ROWS]-cell camera chromaticity "r,g,b" (unit length, "-" for no valid cell), for checking against RAW. */
    fun gridString(s: MeterStats): String = gridCells { gx, gy ->
        meanRgb(s.camRgb, s.valid, gx * COLS / GRID_COLS, (gx + 1) * COLS / GRID_COLS, gy * ROWS / GRID_ROWS, (gy + 1) * ROWS / GRID_ROWS)
    }

    /**
     * The same grid on a RAW frame: black level off, the HAL's lens shading gains on (the ISP
     * applied them to the analysis stream too), clipped quads skipped.
     */
    fun rawGridString(frame: RawFrame, meta: RawMetadata): String {
        val w = frame.width
        val h = frame.height
        val white = meta.whiteLevel * 0.98f
        val lsm = meta.lensShading
        val pos = when (meta.cfa) {
            CfaPattern.GRBG -> intArrayOf(1, 0, 3, 2)
            CfaPattern.GBRG -> intArrayOf(2, 0, 3, 1)
            CfaPattern.BGGR -> intArrayOf(3, 1, 2, 0)
            else -> intArrayOf(0, 1, 2, 3)
        }
        // pos[i]: quad index (row-major in the 2x2 tile) of channel i in R, Geven, Godd, B order
        val sums = Array(GRID_COLS * GRID_ROWS) { DoubleArray(3) }
        val stride = 32
        for (qy in 0 until h / 2 step stride) for (qx in 0 until w / 2 step stride) {
            val x = qx * 2
            val y = qy * 2
            val v = FloatArray(4)
            var clipped = false
            for (i in 0 until 4) {
                val q = pos[i]
                val s = frame.sample(x + (q and 1), y + (q shr 1)).toFloat()
                if (s >= white) clipped = true
                v[i] = s - meta.blackLevel[q]
            }
            if (clipped) continue
            val gx = x * GRID_COLS / w
            val gy = y * GRID_ROWS / h
            val acc = sums[gy * GRID_COLS + gx]
            val fx = (x + 1f) / w
            val fy = (y + 1f) / h
            fun gain(ch: Int): Float {
                if (lsm == null) return 1f
                val cx = fx * (lsm.columnCount - 1)
                val cy = fy * (lsm.rowCount - 1)
                val x0 = cx.toInt().coerceAtMost(lsm.columnCount - 2)
                val y0 = cy.toInt().coerceAtMost(lsm.rowCount - 2)
                val ax = cx - x0
                val ay = cy - y0
                return (lsm.getGainFactor(ch, x0, y0) * (1 - ax) * (1 - ay) + lsm.getGainFactor(ch, x0 + 1, y0) * ax * (1 - ay) +
                    lsm.getGainFactor(ch, x0, y0 + 1) * (1 - ax) * ay + lsm.getGainFactor(ch, x0 + 1, y0 + 1) * ax * ay)
            }
            acc[0] += v[0] * gain(0)
            acc[1] += (v[1] * gain(1) + v[2] * gain(2)) / 2
            acc[2] += v[3] * gain(3)
        }
        return gridCells { gx, gy ->
            val a = sums[gy * GRID_COLS + gx]
            if (a[0] <= 0 || a[1] <= 0 || a[2] <= 0) null else unit(floatArrayOf(a[0].toFloat(), a[1].toFloat(), a[2].toFloat()))
        }
    }

    private fun gridCells(cell: (Int, Int) -> FloatArray?): String = buildString {
        for (gy in 0 until GRID_ROWS) for (gx in 0 until GRID_COLS) {
            if (isNotEmpty()) append(' ')
            val c = cell(gx, gy)
            append(if (c == null) "-" else String.format(Locale.US, "%.4f,%.4f,%.4f", c[0], c[1], c[2]))
        }
    }

    private val SRGB_INVERSE = FloatArray(256) { i ->
        val e = i / 255f
        if (e <= 0.04045f) e / 12.92f else ((e + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat()
    }

    /** Encoded value (index / 255) to linear input, through the curve's points read backwards. */
    private fun inverseLut(curve: TonemapCurve, channel: Int): FloatArray {
        val n = curve.getPointCount(channel)
        val pts = Array(n) { curve.getPoint(channel, it) }.sortedBy { it.y }
        return FloatArray(256) { i ->
            val o = i / 255f
            val hi = pts.indexOfFirst { it.y >= o }
            when {
                hi <= 0 -> pts.first().x
                else -> {
                    val a = pts[hi - 1]
                    val b = pts[hi]
                    if (b.y - a.y < 1e-6f) b.x else a.x + (b.x - a.x) * (o - a.y) / (b.y - a.y)
                }
            }
        }
    }

    private fun invert(m: Array<DoubleArray>): Array<DoubleArray>? {
        val det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        if (kotlin.math.abs(det) < 1e-9) return null
        val c = Array(3) { DoubleArray(3) }
        for (i in 0 until 3) for (j in 0 until 3) {
            val a = m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3] - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3]
            c[i][j] = a / det
        }
        return c
    }
}
