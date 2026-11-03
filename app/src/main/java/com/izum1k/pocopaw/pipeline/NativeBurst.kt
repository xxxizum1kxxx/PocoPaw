package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import com.izum1k.pocopaw.camera.CfaPattern
import java.io.Closeable
import java.nio.ByteBuffer
import kotlin.math.min

/**
 * Finishing parameters of the native pipeline (`LookParams` in `cpp/burstpipe/finish.h`).
 * [chromaDenoise] < 0 scales with the measured noise; [compression] 0 picks the ratio from the scene.
 * A [style] other than "default" ("pixel", "night") replaces every other field with that preset.
 */
data class BurstLook(
    val style: String = "default",
    val lscLumaStrength: Float = 0.85f,
    val chromaDenoise: Float = -1f,
    val compression: Float = 0f,
    val targetKey: Float = 0.23f,
    val contrast: Float = 0.22f,
    val saturation: Float = 1.12f,
    val sharpen: Float = 0.8f,
    val sharpenRadius: Float = 1f,
    val localContrast: Float = 0.3f,
) {
    internal fun toArray() =
        floatArrayOf(lscLumaStrength, chromaDenoise, compression, targetKey, contrast, saturation, sharpen, sharpenRadius, localContrast)
}

/** Demosaic of the classic merge; RCD is the default (pipeline/out/p1/RESULTS.md). */
enum class DemosaicMethod { MHC, RCD, AMAZE }

/**
 * Lateral chromatic aberration of one lens (tools/lca_calib.py): red / blue are sampled at
 * p + (p − c)(k1 + k2 (r / rmax)²). Only the kernel-regression merge applies it.
 */
data class LensLca(val kr1: Float, val kr2: Float, val kb1: Float, val kb2: Float)

/**
 * Per-ISO sensor noise, variance = a · signal + b in raw DN (tools/noise_calib.py curve); linear in ISO between
 * the points. With one, the native session skips estimating the noise from the burst.
 */
class NoiseCurve(val iso: FloatArray, val a: FloatArray, val b: FloatArray)

/**
 * How [NativeBurst.merge] combines the frames (`BurstOptions` in `cpp/burstpipe/burst.h`).
 * [robustLo] / [robustHi] 0 pick the ramp that matches the noise source (calibrated or estimated).
 */
data class MergeOptions(
    val kernelRegression: Boolean = false,
    val robustLo: Float = 0f,
    val robustHi: Float = 0f,
    val motionDilate: Int = 1,
    val demosaic: DemosaicMethod = DemosaicMethod.RCD,
    /** Frequency-domain Wiener merge (classic path); the night mode uses it with spatial shrinkage 2. */
    val frequencyMerge: Boolean = false,
    val frequencyC: Float = 64f,
    val frequencySpatial: Float = 2f,
    val lca: LensLca? = null,
) {
    internal fun toArray() = floatArrayOf(
        if (kernelRegression) 1f else 0f, robustLo, robustHi, motionDilate.toFloat(), demosaic.ordinal.toFloat(),
        if (frequencyMerge) 1f else 0f, frequencyC, frequencySpatial,
        lca?.kr1 ?: 0f, lca?.kr2 ?: 0f, lca?.kb1 ?: 0f, lca?.kb2 ?: 0f,
    )
}

/**
 * One burst inside `libburstpipe`: align, merge, finish and render run on the calling thread and
 * use every core. The frames' direct buffers are read in place, so [burst] must stay reachable
 * until [merge] returns.
 */
class NativeBurst(private val burst: RawBurst) : Closeable {
    private var handle: Long

    var width = 0
        private set
    var height = 0
        private set

    init {
        val meta = burst.metadata
        val frame = burst.reference
        val cfa = when (meta.cfa) {
            CfaPattern.RGGB -> 0
            CfaPattern.GRBG -> 1
            CfaPattern.GBRG -> 2
            CfaPattern.BGGR, CfaPattern.UNKNOWN -> 3
        }
        val black = FloatArray(4)
        for (pos in 0 until 4) black[channelAt(cfa, pos % 2, pos / 2)] = meta.blackLevel[pos]
        val wb = meta.wbGains
        val ccm = FloatArray(9) { i -> meta.colorTransform?.getElement(i % 3, i / 3)?.toFloat() ?: if (i % 4 == 0) 1f else 0f }
        val lsc = meta.lensShading
        val lscGains = lsc?.let { FloatArray(it.gainFactorCount).also { g -> it.copyGainFactors(g, 0) } }
        val array = FloatArray(META_SIZE).apply {
            this[0] = frame.width.toFloat()
            this[1] = frame.height.toFloat()
            this[2] = cfa.toFloat()
            this[3] = meta.whiteLevel.toFloat()
            black.copyInto(this, 4)
            this[8] = wb?.red ?: 1f
            this[9] = wb?.greenEven ?: 1f
            this[10] = wb?.greenOdd ?: 1f
            this[11] = wb?.blue ?: 1f
            ccm.copyInto(this, 12)
            this[21] = (lsc?.rowCount ?: 0).toFloat()
            this[22] = (lsc?.columnCount ?: 0).toFloat()
        }
        val buffers = Array<ByteBuffer>(burst.size) { burst.frames[it].data }
        val exposures = LongArray(burst.size) { burst.frames[it].exposureNs ?: 0L }
        val isos = IntArray(burst.size) { burst.frames[it].iso ?: 100 }
        handle = nativeCreate(array, lscGains, buffers, burst.referenceIndex, exposures, isos)
    }

    /** Before [align]: run the steps on the GPU (OpenCL) where possible, each falling back to the CPU on failure. */
    fun useGpu(enabled: Boolean) = nativeUseGpu(live(), enabled)

    /** Before [align]: use the camera's calibrated noise instead of estimating it from the burst. */
    fun setNoiseCurve(curve: NoiseCurve) = nativeSetNoiseCurve(live(), curve.iso, curve.a, curve.b)

    /** Before [align]: process only the centre 1/[zoom]; [finish] scales the result back to the sensor size. */
    fun crop(zoom: Float) = nativeCrop(live(), zoom)

    fun align() = nativeAlign(live())

    fun merge(options: MergeOptions) = nativeMerge(live(), options.toArray())

    /** [keepHdr] keeps the scene-linear luminance [encodeUltraHdr] needs. */
    fun finish(look: BurstLook, keepHdr: Boolean = false) {
        val size = nativeFinish(live(), look.style, look.toArray(), keepHdr)
        width = (size ushr 32).toInt()
        height = (size and 0xFFFFFFFFL).toInt()
    }

    /** The finished image in the sensor's orientation. */
    fun render(): Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { nativeRender(live(), it) }

    /** The finished image as an Ultra HDR JPEG (SDR base + gain map up to [headroom]× SDR white), or null. */
    fun encodeUltraHdr(quality: Int = 95, headroom: Float = 4f): ByteArray? = nativeEncodeUltraHdr(live(), quality, headroom)

    fun summary(): String = nativeSummary(live())

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
        }
    }

    private fun live(): Long = handle.also { check(it != 0L) { "NativeBurst already closed" } }

    companion object {
        private const val META_SIZE = 23

        /**
         * Points the OpenCL program cache at [cacheDir] and creates the GPU device, compiling the kernels
         * on the first run (~0.5 s; call off the main thread, before the first GPU burst): the device's
         * properties, one per line, or why there is no usable GPU.
         */
        fun gpuInit(cacheDir: String): String = nativeGpuInit(cacheDir)

        /** Read-only findings about the Hexagon cDSP (FastRPC capabilities, unsigned protection domain), one per line. */
        fun dspProbe(): String = nativeDspProbe()

        init {
            System.loadLibrary("burstpipe")
        }

        /** Bayer channel (0 R, 1 G on even rows, 2 G on odd rows, 3 B) at a 2×2 position, as in `cfa_channel`. */
        private fun channelAt(cfa: Int, x: Int, y: Int): Int {
            val layouts = arrayOf(intArrayOf(0, 1, 2, 3), intArrayOf(1, 0, 3, 2), intArrayOf(1, 3, 0, 2), intArrayOf(3, 1, 2, 0))
            val ch = layouts[cfa][(y and 1) * 2 + (x and 1)]
            return if (ch == 1 || ch == 2) (if (y and 1 == 1) 2 else 1) else ch
        }

        @JvmStatic private external fun nativeCreate(
            meta: FloatArray, lsc: FloatArray?, frames: Array<ByteBuffer>, ref: Int, exposureNs: LongArray, iso: IntArray,
        ): Long
        @JvmStatic private external fun nativeCrop(handle: Long, zoom: Float)
        @JvmStatic private external fun nativeUseGpu(handle: Long, enabled: Boolean)
        /**
         * The frame to merge onto: a bracket's shortest exposure (it keeps the highlights), otherwise the sharpest
         * of the first three frames by gradient energy (a handheld burst's first frame is often shaken by the press).
         */
        fun referenceFor(burst: RawBurst): Int {
            if (burst.size < 2) return 0
            val expo = burst.frames.map { (it.exposureNs ?: 0L).toDouble() * (it.iso ?: 100) }
            val shortest = expo.indices.minBy { expo[it] }
            if (expo[shortest] < expo.max() * 0.9) return shortest
            val f = burst.frames[0]
            return nativeSharpestFrame(Array(min(3, burst.size)) { burst.frames[it].data }, f.width, f.height)
        }

        @JvmStatic private external fun nativeSharpestFrame(frames: Array<ByteBuffer>, width: Int, height: Int): Int
        @JvmStatic private external fun nativeSetNoiseCurve(handle: Long, iso: FloatArray, a: FloatArray, b: FloatArray)
        @JvmStatic private external fun nativeAlign(handle: Long)
        @JvmStatic private external fun nativeMerge(handle: Long, options: FloatArray)
        @JvmStatic private external fun nativeFinish(handle: Long, style: String, look: FloatArray, keepHdr: Boolean): Long
        @JvmStatic private external fun nativeEncodeUltraHdr(handle: Long, quality: Int, headroom: Float): ByteArray?
        @JvmStatic private external fun nativeRender(handle: Long, bitmap: Bitmap)
        @JvmStatic private external fun nativeSummary(handle: Long): String
        @JvmStatic private external fun nativeRelease(handle: Long)
        @JvmStatic private external fun nativeGpuInit(cacheDir: String): String
        @JvmStatic private external fun nativeDspProbe(): String
    }
}
