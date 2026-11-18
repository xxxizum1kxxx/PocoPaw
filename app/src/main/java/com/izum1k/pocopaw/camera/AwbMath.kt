package com.izum1k.pocopaw.camera

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin radix-2 Cooley-Tukey 2D FFT for 64x64 log-chroma grids (used by FFCC and C5 AWB).
 */
object Fft2d64 {
    const val N = 64
    const val SIZE = N * N
    private const val LOG2_N = 6

    private val BIT_REV = IntArray(N) { i ->
        var x = i
        var r = 0
        for (b in 0 until LOG2_N) {
            r = (r shl 1) or (x and 1)
            x = x ushr 1
        }
        r
    }

    // Twiddle factors per stage (len = 2, 4, 8, 16, 32, 64)
    private val COS_TAB = DoubleArray(N / 2) { k -> cos(-2.0 * PI * k / N) }
    private val SIN_TAB = DoubleArray(N / 2) { k -> sin(-2.0 * PI * k / N) }

    fun fft1d(re: DoubleArray, im: DoubleArray, offset: Int, stride: Int) {
        for (i in 0 until N) {
            val j = BIT_REV[i]
            if (i < j) {
                val ai = offset + i * stride
                val aj = offset + j * stride
                val tr = re[ai]; re[ai] = re[aj]; re[aj] = tr
                val ti = im[ai]; im[ai] = im[aj]; im[aj] = ti
            }
        }
        var half = 1
        var step = N / 2
        while (half < N) {
            val len = half shl 1
            for (start in 0 until N step len) {
                for (k in 0 until half) {
                    val twIdx = k * step
                    val wr = COS_TAB[twIdx]
                    val wi = SIN_TAB[twIdx]
                    val idx0 = offset + (start + k) * stride
                    val idx1 = idx0 + half * stride
                    val xr = re[idx1]
                    val xi = im[idx1]
                    val tr = wr * xr - wi * xi
                    val ti = wr * xi + wi * xr
                    re[idx1] = re[idx0] - tr
                    im[idx1] = im[idx0] - ti
                    re[idx0] += tr
                    im[idx0] += ti
                }
            }
            half = len
            step = step ushr 1
        }
    }

    fun fft2(re: DoubleArray, im: DoubleArray) {
        require(re.size == SIZE && im.size == SIZE)
        for (r in 0 until N) fft1d(re, im, r * N, 1)
        for (c in 0 until N) fft1d(re, im, c, N)
    }

    fun ifft2(re: DoubleArray, im: DoubleArray) {
        require(re.size == SIZE && im.size == SIZE)
        for (i in 0 until SIZE) im[i] = -im[i]
        fft2(re, im)
        val inv = 1.0 / SIZE
        for (i in 0 until SIZE) {
            re[i] *= inv
            im[i] = -im[i] * inv
        }
    }

    fun unitRgb(r: Double, g: Double, b: Double): FloatArray? {
        if (!r.isFinite() || !g.isFinite() || !b.isFinite()) return null
        val norm = sqrt(r * r + g * g + b * b)
        if (!norm.isFinite() || norm <= 1e-12) return null
        return floatArrayOf((r / norm).toFloat(), (g / norm).toFloat(), (b / norm).toFloat())
    }
}
