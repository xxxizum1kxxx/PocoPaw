package com.izum1k.pocopaw.camera

import android.content.res.AssetManager
import android.hardware.camera2.params.RggbChannelVector
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max

/**
 * Fast Fourier Color Constancy (Barron & Tsai, CVPR 2017) illuminant estimator matching
 * `pipeline/tools/ffcc.py`. Uses per-camera calibration (`calib/ffcc_cam{0,2}.bin`).
 */
class FfccAwbEstimator(
    private val models: Map<String, FfccModel>,
    val usePrior: Boolean = false,
) : AwbEstimator {

    class FfccModel(
        val n: Int,
        val u0: Double,
        val v0: Double,
        val bin: Double,
        val f0Re: DoubleArray,
        val f0Im: DoubleArray,
        val f1Re: DoubleArray,
        val f1Im: DoubleArray,
        val bias: FloatArray,
    ) {
        companion object {
            private const val MAGIC = 0x46464343 // "FFCC"

            fun fromBytes(bytes: ByteArray): FfccModel {
                val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                val magic = buf.int
                require(magic == MAGIC) { "Invalid FFCC magic: 0x${magic.toString(16)}" }
                val n = buf.int
                require(n == Fft2d64.N) { "Expected FFCC N=${Fft2d64.N}, got $n" }
                val u0 = buf.double
                val v0 = buf.double
                val bin = buf.double
                val sz = n * n
                val f0Re = DoubleArray(sz) { buf.float.toDouble() }
                val f1Re = DoubleArray(sz) { buf.float.toDouble() }
                val f0Im = DoubleArray(sz)
                val f1Im = DoubleArray(sz)
                Fft2d64.fft2(f0Re, f0Im)
                Fft2d64.fft2(f1Re, f1Im)
                val bias = FloatArray(sz) { buf.float }
                return FfccModel(n, u0, v0, bin, f0Re, f0Im, f1Re, f1Im, bias)
            }
        }
    }

    override fun estimate(stats: MeterStats, cameraId: String?): RggbChannelVector? =
        estimateIlluminant(stats, cameraId)?.let { SceneMeter.gainsFor(it) }

    fun estimateIlluminant(
        stats: MeterStats,
        cameraId: String?,
        usePrior: Boolean = this.usePrior,
    ): FloatArray? = estimateIlluminant(
        rgb = stats.camRgb,
        valid = stats.valid,
        width = SceneMeter.COLS,
        height = SceneMeter.ROWS,
        cameraId = cameraId,
        usePrior = usePrior,
    )

    fun estimateIlluminant(
        rgb: FloatArray,
        valid: BooleanArray?,
        width: Int,
        height: Int,
        cameraId: String?,
        usePrior: Boolean = this.usePrior,
    ): FloatArray? {
        val model = cameraId?.let { models[it] } ?: return null
        val nPix = width * height
        if (width < 2 || height < 2 || rgb.size < nPix * 3) return null
        if (valid != null && valid.size < nPix) return null

        val h0Re = DoubleArray(Fft2d64.SIZE)
        val h1Re = DoubleArray(Fft2d64.SIZE)
        val dev = localDev(rgb, width, height)

        var count0 = 0
        var count1 = 0
        val n = model.n
        val invBin = 1.0 / model.bin
        for (i in 0 until nPix) {
            if (valid != null && !valid[i]) continue
            val base = i * 3
            val r0 = rgb[base].toDouble()
            val g0 = rgb[base + 1].toDouble()
            val b0 = rgb[base + 2].toDouble()
            if (r0 > MIN_CHROMA && g0 > MIN_CHROMA && b0 > MIN_CHROMA) {
                val u = ln(g0 / r0)
                val v = ln(g0 / b0)
                val iu = floor((u - model.u0) * invBin).toInt().mod(n)
                val iv = floor((v - model.v0) * invBin).toInt().mod(n)
                h0Re[iu * n + iv] += 1.0
                count0++
            }
            val r1 = dev[base].toDouble()
            val g1 = dev[base + 1].toDouble()
            val b1 = dev[base + 2].toDouble()
            if (r1 > MIN_CHROMA && g1 > MIN_CHROMA && b1 > MIN_CHROMA) {
                val u = ln(g1 / r1)
                val v = ln(g1 / b1)
                val iu = floor((u - model.u0) * invBin).toInt().mod(n)
                val iv = floor((v - model.v0) * invBin).toInt().mod(n)
                h1Re[iu * n + iv] += 1.0
                count1++
            }
        }
        if (count0 == 0) return null

        val s0 = max(count0.toDouble(), 1e-9)
        val s1 = max(count1.toDouble(), 1e-9)
        for (i in 0 until Fft2d64.SIZE) {
            h0Re[i] /= s0
            h1Re[i] /= s1
        }
        val h0Im = DoubleArray(Fft2d64.SIZE)
        val h1Im = DoubleArray(Fft2d64.SIZE)
        Fft2d64.fft2(h0Re, h0Im)
        Fft2d64.fft2(h1Re, h1Im)

        val convRe = DoubleArray(Fft2d64.SIZE)
        val convIm = DoubleArray(Fft2d64.SIZE)
        for (k in 0 until Fft2d64.SIZE) {
            convRe[k] = (h0Re[k] * model.f0Re[k] - h0Im[k] * model.f0Im[k]) +
                (h1Re[k] * model.f1Re[k] - h1Im[k] * model.f1Im[k])
            convIm[k] = (h0Re[k] * model.f0Im[k] + h0Im[k] * model.f0Re[k]) +
                (h1Re[k] * model.f1Im[k] + h1Im[k] * model.f1Re[k])
        }
        Fft2d64.ifft2(convRe, convIm)

        var maxL = Double.NEGATIVE_INFINITY
        val logits = DoubleArray(Fft2d64.SIZE)
        for (k in 0 until Fft2d64.SIZE) {
            val v = convRe[k] + if (usePrior) model.bias[k].toDouble() else 0.0
            logits[k] = v
            if (v > maxL) maxL = v
        }
        var sumP = 0.0
        var bestIdx = 0
        var bestP = -1.0
        val p = DoubleArray(Fft2d64.SIZE)
        for (k in 0 until Fft2d64.SIZE) {
            val e = exp(logits[k] - maxL)
            p[k] = e
            sumP += e
            if (e > bestP) {
                bestP = e
                bestIdx = k
            }
        }
        if (sumP <= 0.0 || !sumP.isFinite()) return null
        val invSum = 1.0 / sumP
        val rowSum = DoubleArray(n)
        val colSum = DoubleArray(n)
        for (iu in 0 until n) {
            val rowOff = iu * n
            var rs = 0.0
            for (iv in 0 until n) {
                val pv = p[rowOff + iv] * invSum
                rs += pv
                colSum[iv] += pv
            }
            rowSum[iu] = rs
        }

        val modeU = bestIdx / n
        val modeV = bestIdx % n
        val half = n / 2
        var duSum = 0.0
        var dvSum = 0.0
        for (g in 0 until n) {
            val du = (g - modeU + half).mod(n) - half
            val dv = (g - modeV + half).mod(n) - half
            duSum += rowSum[g] * du
            dvSum += colSum[g] * dv
        }
        val fu = modeU + duSum
        val fv = modeV + dvSum
        val u = model.u0 + (fu + 0.5) * model.bin
        val v = model.v0 + (fv + 0.5) * model.bin
        return Fft2d64.unitRgb(exp(-u), 1.0, exp(-v))
    }

    private fun localDev(rgb: FloatArray, width: Int, height: Int): FloatArray {
        val out = FloatArray(width * height * 3)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val cOff = (y * width + x) * 3
                val r = rgb[cOff]
                val g = rgb[cOff + 1]
                val b = rgb[cOff + 2]
                var ar = 0f
                var ag = 0f
                var ab = 0f
                for (dy in -1..1) {
                    val ny = reflectIndex(y + dy, height)
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = reflectIndex(x + dx, width)
                        val nOff = (ny * width + nx) * 3
                        ar += abs(r - rgb[nOff])
                        ag += abs(g - rgb[nOff + 1])
                        ab += abs(b - rgb[nOff + 2])
                    }
                }
                out[cOff] = ar / 8f
                out[cOff + 1] = ag / 8f
                out[cOff + 2] = ab / 8f
            }
        }
        return out
    }

    private fun reflectIndex(i: Int, n: Int): Int = when {
        n <= 1 -> 0
        i < 0 -> -i
        i >= n -> 2 * n - 2 - i
        else -> i
    }

    companion object {
        private const val MIN_CHROMA = 1e-4
        private val CAMERA_IDS = listOf("0", "2")

        fun fromAssets(assets: AssetManager, usePrior: Boolean = false): FfccAwbEstimator {
            val models = LinkedHashMap<String, FfccModel>()
            for (camId in CAMERA_IDS) {
                runCatching {
                    assets.open("calib/ffcc_cam$camId.bin").use { input ->
                        models[camId] = FfccModel.fromBytes(input.readBytes())
                    }
                }
            }
            return FfccAwbEstimator(models, usePrior)
        }

        fun fromFiles(filesByCamId: Map<String, File>, usePrior: Boolean = false): FfccAwbEstimator {
            val models = LinkedHashMap<String, FfccModel>()
            for ((camId, file) in filesByCamId) {
                if (file.isFile) {
                    runCatching { models[camId] = FfccModel.fromBytes(file.readBytes()) }
                }
            }
            return FfccAwbEstimator(models, usePrior)
        }
    }
}
