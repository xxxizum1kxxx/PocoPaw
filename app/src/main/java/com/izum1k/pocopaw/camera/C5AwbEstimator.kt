package com.izum1k.pocopaw.camera

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.hardware.camera2.params.RggbChannelVector
import com.izum1k.pocopaw.pipeline.AccelBackend
import com.izum1k.pocopaw.pipeline.OrtConfig
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * C5 (Afifi et al., ICCV 2021) learned cross-camera AWB estimator matching
 * `pipeline/tools/awb_c5.py` and `third_party/C5/src/ops.py` exact conventions.
 */
class C5AwbEstimator private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
    private val outputName: String,
) : AwbEstimator, AutoCloseable {

    private val extrasByCamera = HashMap<String, ArrayDeque<FloatArray>>()

    override fun estimate(stats: MeterStats, cameraId: String?): RggbChannelVector? =
        estimateIlluminant(stats, cameraId)?.let { SceneMeter.gainsFor(it) }

    fun estimateIlluminant(
        stats: MeterStats,
        cameraId: String?,
        extras: List<FloatArray>? = null,
    ): FloatArray? = estimateIlluminant(
        rgb = stats.camRgb,
        valid = stats.valid,
        width = SceneMeter.COLS,
        height = SceneMeter.ROWS,
        cameraId = cameraId,
        extras = extras,
    )

    fun estimateIlluminant(
        rgb: FloatArray,
        valid: BooleanArray?,
        width: Int,
        height: Int,
        cameraId: String?,
        extras: List<FloatArray>? = null,
    ): FloatArray? {
        val nPix = width * height
        if (width < 2 || height < 2 || rgb.size < nPix * 3) return null
        if (valid != null && valid.size < nPix) return null

        val hq = computeHistogram(rgb, valid, width, height) ?: return null
        val pool: List<FloatArray> = if (extras != null) {
            extras.ifEmpty { listOf(hq) }
        } else {
            val camKey = cameraId ?: "_default"
            synchronized(extrasByCamera) {
                val deque = extrasByCamera.getOrPut(camKey) { ArrayDeque(M - 1) }
                val snapshot = if (deque.isEmpty()) listOf(hq) else deque.toList()
                val last = deque.lastOrNull()
                if (last == null || l2Distance(hq, last) > EXTRA_DIVERSITY_THRESHOLD) {
                    if (deque.size >= M - 1) deque.removeFirst()
                    deque.addLast(hq)
                }
                snapshot
            }
        }

        val inputBuf = FloatArray(M * 4 * HIST_SIZE)
        for (slot in 0 until M) {
            val h = if (slot == 0) hq else pool[(slot - 1) % pool.size]
            val slotBase = slot * 4 * HIST_SIZE
            System.arraycopy(h, 0, inputBuf, slotBase, HIST_SIZE)
            System.arraycopy(h, HIST_SIZE, inputBuf, slotBase + HIST_SIZE, HIST_SIZE)
            System.arraycopy(U_NORM, 0, inputBuf, slotBase + 2 * HIST_SIZE, HIST_SIZE)
            System.arraycopy(V_NORM, 0, inputBuf, slotBase + 3 * HIST_SIZE, HIST_SIZE)
        }

        val outFb = runCatching {
            val shape = longArrayOf(1L, (M * 4).toLong(), N_BINS.toLong(), N_BINS.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(inputBuf), shape).use { tensor ->
                session.run(mapOf(inputName to tensor), setOf(outputName)).use { result ->
                    val fbTensor = result[0] as OnnxTensor
                    val fb = FloatArray(3 * HIST_SIZE)
                    fbTensor.floatBuffer.get(fb)
                    fb
                }
            }
        }.getOrNull() ?: return null

        val n0Re = DoubleArray(HIST_SIZE) { hq[it].toDouble() }
        val n0Im = DoubleArray(HIST_SIZE)
        val n1Re = DoubleArray(HIST_SIZE) { hq[HIST_SIZE + it].toDouble() }
        val n1Im = DoubleArray(HIST_SIZE)
        val f0Re = DoubleArray(HIST_SIZE) { outFb[it].toDouble() }
        val f0Im = DoubleArray(HIST_SIZE)
        val f1Re = DoubleArray(HIST_SIZE) { outFb[HIST_SIZE + it].toDouble() }
        val f1Im = DoubleArray(HIST_SIZE)

        Fft2d64.fft2(n0Re, n0Im)
        Fft2d64.fft2(n1Re, n1Im)
        Fft2d64.fft2(f0Re, f0Im)
        Fft2d64.fft2(f1Re, f1Im)

        val convRe = DoubleArray(HIST_SIZE)
        val convIm = DoubleArray(HIST_SIZE)
        for (k in 0 until HIST_SIZE) {
            convRe[k] = (n0Re[k] * f0Re[k] - n0Im[k] * f0Im[k]) +
                (n1Re[k] * f1Re[k] - n1Im[k] * f1Im[k])
            convIm[k] = (n0Re[k] * f0Im[k] + n0Im[k] * f0Re[k]) +
                (n1Re[k] * f1Im[k] + n1Im[k] * f1Re[k])
        }
        Fft2d64.ifft2(convRe, convIm)

        val bOff = 2 * HIST_SIZE
        val logits = DoubleArray(HIST_SIZE)
        var maxL = Double.NEGATIVE_INFINITY
        for (k in 0 until HIST_SIZE) {
            val v = (convRe[k] + outFb[bOff + k].toDouble()).coerceIn(-100.0, 100.0)
            logits[k] = v
            if (v > maxL) maxL = v
        }

        val p = DoubleArray(HIST_SIZE)
        var sumP = 0.0
        for (k in 0 until HIST_SIZE) {
            val e = exp(logits[k] - maxL)
            p[k] = e
            sumP += e
        }
        if (sumP <= 0.0 || !sumP.isFinite()) return null

        var uCoord = 0.0
        var vCoord = 0.0
        val invSum = 1.0 / sumP
        for (row in 0 until N_BINS) {
            val vc = 31.5 - row
            val rOff = row * N_BINS
            for (col in 0 until N_BINS) {
                val uc = col - 31.5
                val pv = p[rOff + col] * invSum
                uCoord += pv * uc
                vCoord += pv * vc
            }
        }
        // ops.from_coord_to_uv uses (2 * BOUND) / N_BINS = 5.70 / 64.0
        val uVal = uCoord * UV_SCALE
        val vVal = vCoord * UV_SCALE
        return Fft2d64.unitRgb(exp(-uVal), 1.0, exp(-vVal))
    }

    /**
     * Computes `[N0; N1]` (`2 * 64 * 64` float array, row-major `[channel, row, col]`),
     * matching `awb_c5.histogram` and `ops.py`. Returns `null` if no valid pixels exist.
     */
    fun computeHistogram(
        rgb: FloatArray,
        valid: BooleanArray?,
        width: Int,
        height: Int,
    ): FloatArray? {
        val nPix = width * height
        if (width < 2 || height < 2 || rgb.size < nPix * 3) return null
        if (valid != null && valid.size < nPix) return null

        val out = FloatArray(2 * HIST_SIZE)
        val n0 = DoubleArray(HIST_SIZE)
        val n1 = DoubleArray(HIST_SIZE)
        var validCount = 0

        for (i in 0 until nPix) {
            if (valid != null && !valid[i]) continue
            val off = i * 3
            val r = rgb[off].toDouble()
            val g = rgb[off + 1].toDouble()
            val b = rgb[off + 2].toDouble()
            if (r + g + b <= EPS) continue
            validCount++
            accumulatePixel(r, g, b, n0)
        }
        if (validCount == 0) return null

        val edges = computeEdges(rgb, width, height)
        for (i in 0 until nPix) {
            if (valid != null && !valid[i]) continue
            val off = i * 3
            val r = edges[off].toDouble()
            val g = edges[off + 1].toDouble()
            val b = edges[off + 2].toDouble()
            if (r + g + b <= EPS) continue
            accumulatePixel(r, g, b, n1)
        }

        val sum0 = n0.sum() + EPS
        val sum1 = n1.sum() + EPS
        for (k in 0 until HIST_SIZE) {
            out[k] = sqrt(n0[k] / sum0).toFloat()
            out[HIST_SIZE + k] = sqrt(n1[k] / sum1).toFloat()
        }
        return out
    }

    private fun accumulatePixel(r: Double, g: Double, b: Double, hist: DoubleArray) {
        val logG = ln(g + EPS)
        val u = logG - ln(r + EPS)
        val v = logG - ln(b + EPS)
        val iy = sqrt(r * r + g * g + b * b)

        val iuApprox = floor((u + BOUND) / HIST_EPS).toInt()
        val ivApprox = floor((BOUND - v) / HIST_EPS).toInt()
        val iu0 = (iuApprox - 1).coerceAtLeast(0)
        val iu1 = (iuApprox + 1).coerceAtMost(N_BINS - 1)
        val iv0 = (ivApprox - 1).coerceAtLeast(0)
        val iv1 = (ivApprox + 1).coerceAtMost(N_BINS - 1)
        if (iu0 > iu1 || iv0 > iv1) return

        for (iv in iv0..iv1) {
            val dv = abs(v - A_V[iv])
            if (dv <= 0.0 || dv > HIST_EPS) continue
            val rOff = iv * N_BINS
            for (iu in iu0..iu1) {
                val du = abs(u - A_U[iu])
                if (du <= 0.0 || du > HIST_EPS) continue
                hist[rOff + iu] += iy
            }
        }
    }

    /** Matches `ops.compute_edges`: `cv2.BORDER_REFLECT` clamps `-1 -> 0` and `W -> W-1`. */
    private fun computeEdges(rgb: FloatArray, width: Int, height: Int): FloatArray {
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
                    val ny = (y + dy).coerceIn(0, height - 1)
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = (x + dx).coerceIn(0, width - 1)
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

    private fun l2Distance(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) {
            val d = a[i] - b[i]
            s += d * d
        }
        return sqrt(s)
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        private const val N_BINS = 64
        private const val HIST_SIZE = N_BINS * N_BINS
        private const val M = 7
        private const val BOUND = 2.85
        private const val EPS = 1e-9
        private const val HIST_EPS = (2.0 * BOUND) / (N_BINS - 1)
        private const val UV_SCALE = (2.0 * BOUND) / N_BINS
        private const val EXTRA_DIVERSITY_THRESHOLD = 0.05f

        private val A_U = DoubleArray(N_BINS) { iu -> -BOUND + iu * HIST_EPS }
        private val A_V = DoubleArray(N_BINS) { iv -> BOUND - iv * HIST_EPS }

        private val U_NORM = FloatArray(HIST_SIZE) { k -> (k % N_BINS) / 63.0f }
        private val V_NORM = FloatArray(HIST_SIZE) { k -> (63 - (k / N_BINS)) / 63.0f }

        fun fromAssets(
            assets: AssetManager,
            backend: AccelBackend = AccelBackend.AUTO,
        ): C5AwbEstimator {
            val bytes = assets.open("models/c5_m7.onnx").use { it.readBytes() }
            return fromBytes(bytes, backend)
        }

        fun fromFile(
            file: File,
            backend: AccelBackend = AccelBackend.AUTO,
        ): C5AwbEstimator = fromBytes(file.readBytes(), backend)

        fun fromBytes(
            modelBytes: ByteArray,
            backend: AccelBackend = AccelBackend.AUTO,
        ): C5AwbEstimator {
            val env = OrtEnvironment.getEnvironment()
            val session = OrtConfig.createSessionOptions(backend = backend).use { opts ->
                env.createSession(modelBytes, opts)
            }
            val inName = session.inputNames.firstOrNull() ?: "hist"
            val outName = session.outputNames.firstOrNull() ?: "fb"
            return C5AwbEstimator(env, session, inName, outName)
        }
    }
}
