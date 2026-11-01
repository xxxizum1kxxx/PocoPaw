package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import com.izum1k.pocopaw.camera.CfaPattern
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * Finishing parameters of the native pipeline (`LookParams` in `cpp/burstpipe/finish.h`).
 * [chromaDenoise] < 0 scales with the measured noise; [compression] 0 picks the ratio from the scene.
 */
data class BurstLook(
    val lscLumaStrength: Float = 0.85f,
    val chromaDenoise: Float = -1f,
    val compression: Float = 0f,
    val targetKey: Float = 0.18f,
    val contrast: Float = 0.22f,
    val saturation: Float = 1.12f,
    val sharpen: Float = 0.6f,
    val sharpenRadius: Float = 1f,
) {
    internal fun toArray() = floatArrayOf(lscLumaStrength, chromaDenoise, compression, targetKey, contrast, saturation, sharpen, sharpenRadius)
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
        handle = nativeCreate(array, lscGains, buffers, burst.referenceIndex)
    }

    fun align() = nativeAlign(live())

    fun merge(kernelRegression: Boolean, robustLo: Float = 2f, robustHi: Float = 4.5f) =
        nativeMerge(live(), kernelRegression, robustLo, robustHi)

    fun finish(look: BurstLook) {
        val size = nativeFinish(live(), look.toArray())
        width = (size ushr 32).toInt()
        height = (size and 0xFFFFFFFFL).toInt()
    }

    /** The finished image in the sensor's orientation. */
    fun render(): Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { nativeRender(live(), it) }

    fun summary(): String = nativeSummary(live())

    override fun close() {
        if (handle != 0L) {
            nativeRelease(handle)
            handle = 0L
        }
    }

    private fun live(): Long = handle.also { check(it != 0L) { "NativeBurst already closed" } }

    private companion object {
        const val META_SIZE = 23

        init {
            System.loadLibrary("burstpipe")
        }

        /** Bayer channel (0 R, 1 G on even rows, 2 G on odd rows, 3 B) at a 2×2 position, as in `cfa_channel`. */
        fun channelAt(cfa: Int, x: Int, y: Int): Int {
            val layouts = arrayOf(intArrayOf(0, 1, 2, 3), intArrayOf(1, 0, 3, 2), intArrayOf(1, 3, 0, 2), intArrayOf(3, 1, 2, 0))
            val ch = layouts[cfa][(y and 1) * 2 + (x and 1)]
            return if (ch == 1 || ch == 2) (if (y and 1 == 1) 2 else 1) else ch
        }

        @JvmStatic external fun nativeCreate(meta: FloatArray, lsc: FloatArray?, frames: Array<ByteBuffer>, ref: Int): Long
        @JvmStatic external fun nativeAlign(handle: Long)
        @JvmStatic external fun nativeMerge(handle: Long, kernelRegression: Boolean, robustLo: Float, robustHi: Float)
        @JvmStatic external fun nativeFinish(handle: Long, look: FloatArray): Long
        @JvmStatic external fun nativeRender(handle: Long, bitmap: Bitmap)
        @JvmStatic external fun nativeSummary(handle: Long): String
        @JvmStatic external fun nativeRelease(handle: Long)
    }
}
