package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.DngCreator
import android.media.ExifInterface
import android.util.Size
import com.izum1k.pocopaw.camera.CfaPattern
import java.nio.ByteOrder
import java.nio.channels.Channels
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * A small sRGB rendering of the reference frame: 2×2 Bayer tiles become one pixel (subsampled to
 * [longSide]), black level, white balance and the HAL's colour transform, then the sRGB curve.
 * Good enough for a thumbnail; it is not the finishing pipeline.
 */
class QuickLookStage(private val longSide: Int = 512) : PipelineStage {
    override val name = "quick_look"

    override suspend fun run(ctx: PipelineContext) {
        val frame = ctx.burst.reference
        val meta = ctx.burst.metadata
        val tilesW = frame.width / 2
        val tilesH = frame.height / 2
        val step = max(1, ceil(max(tilesW, tilesH) / longSide.toFloat()).toInt())
        val outW = tilesW / step
        val outH = tilesH / step

        val layout = channelLayout(meta.cfa)
        val rPos = layout.indexOf('R')
        val bPos = layout.indexOf('B')
        val gPos = layout.withIndex().filter { it.value == 'G' }.map { it.index }
        val range = FloatArray(4) { (meta.whiteLevel - meta.blackLevel[it]).coerceAtLeast(1f) }
        val gains = meta.wbGains
        val gR = gains?.red ?: 1f
        val gG = gains?.let { (it.greenEven + it.greenOdd) / 2f } ?: 1f
        val gB = gains?.blue ?: 1f
        val m = FloatArray(9) { i ->
            meta.colorTransform?.getElement(i % 3, i / 3)?.toFloat() ?: if (i % 4 == 0) 1f else 0f
        }
        val samples = frame.samples()
        val width = frame.width

        fun at(tileX: Int, tileY: Int, pos: Int): Float {
            val x = tileX * 2 + pos % 2
            val y = tileY * 2 + pos / 2
            val v = samples.get(y * width + x).toInt() and 0xFFFF
            return ((v - meta.blackLevel[pos]) / range[pos]).coerceAtLeast(0f)
        }

        val pixels = IntArray(outW * outH)
        for (oy in 0 until outH) {
            val ty = oy * step
            for (ox in 0 until outW) {
                val tx = ox * step
                val r = at(tx, ty, rPos) * gR
                val g = (at(tx, ty, gPos[0]) + at(tx, ty, gPos[1])) * 0.5f * gG
                val b = at(tx, ty, bPos) * gB
                val sr = m[0] * r + m[1] * g + m[2] * b
                val sg = m[3] * r + m[4] * g + m[5] * b
                val sb = m[6] * r + m[7] * g + m[8] * b
                pixels[oy * outW + ox] = (0xFF shl 24) or (encode(sr) shl 16) or (encode(sg) shl 8) or encode(sb)
            }
        }
        val bitmap = Bitmap.createBitmap(pixels, outW, outH, Bitmap.Config.ARGB_8888)
        ctx[PipelineKeys.QuickLook] = bitmap
        ctx[PipelineKeys.QuickLookUpright] = if (meta.orientationDegrees == 0) {
            bitmap
        } else {
            Bitmap.createBitmap(bitmap, 0, 0, outW, outH, Matrix().apply { postRotate(meta.orientationDegrees.toFloat()) }, true)
        }
    }

    private fun encode(linear: Float): Int {
        val c = linear.coerceIn(0f, 1f)
        val s = if (c <= 0.0031308f) 12.92f * c else 1.055f * c.pow(1f / 2.4f) - 0.055f
        return (s * 255f).roundToInt().coerceIn(0, 255)
    }

    private fun channelLayout(cfa: CfaPattern): String = when (cfa) {
        CfaPattern.RGGB -> "RGGB"
        CfaPattern.GRBG -> "GRBG"
        CfaPattern.GBRG -> "GBRG"
        CfaPattern.BGGR, CfaPattern.UNKNOWN -> "BGGR"
    }
}

/** Writes the reference frame unchanged as DNG, with the quick look as its embedded thumbnail. */
class DngWriteStage : PipelineStage {
    override val name = "dng"

    override suspend fun run(ctx: PipelineContext) {
        val frame = ctx.burst.reference
        val meta = ctx.burst.metadata
        val dng = DngCreator(meta.characteristics, frame.result)
        try {
            dng.setOrientation(exifOrientation(meta.orientationDegrees))
            dng.setDescription("PocoPaw")
            ctx[PipelineKeys.QuickLook]?.let { dng.setThumbnail(fitWithin(it, DNG_THUMB_MAX)) }
            val buffer = frame.data.duplicate().order(ByteOrder.nativeOrder())
            buffer.rewind()
            val saved = ctx.sink.write(OutputKind.DNG, MediaStoreWriter.baseName(meta.captureTimeMillis), meta.captureTimeMillis) { out ->
                dng.writeByteBuffer(out, Size(frame.width, frame.height), buffer, 0)
            }
            ctx.addOutput(saved)
        } finally {
            dng.close()
        }
    }

    private fun fitWithin(src: Bitmap, maxSide: Int): Bitmap {
        val scale = maxSide / max(src.width, src.height).toFloat()
        if (scale >= 1f) return src
        return Bitmap.createScaledBitmap(src, (src.width * scale).toInt().coerceAtLeast(1), (src.height * scale).toInt().coerceAtLeast(1), true)
    }

    private fun exifOrientation(degrees: Int): Int = when (degrees) {
        90 -> ExifInterface.ORIENTATION_ROTATE_90
        180 -> ExifInterface.ORIENTATION_ROTATE_180
        270 -> ExifInterface.ORIENTATION_ROTATE_270
        else -> ExifInterface.ORIENTATION_NORMAL
    }

    private companion object {
        const val DNG_THUMB_MAX = 256
    }
}

/** After the merge: the merged mosaic as DNG ([MergedDngWriter]), or the reference frame when there is no mosaic to write. */
class MergedDngStage : PipelineStage {
    override val name = "merged_dng"

    override suspend fun run(ctx: PipelineContext) {
        val meta = ctx.burst.metadata
        val file = MergedDngWriter(meta, ctx.burst.reference.result)
            .encode(ctx.require(PipelineKeys.Native), ctx[PipelineKeys.QuickLook])
            ?: return DngWriteStage().run(ctx)
        try {
            val saved = ctx.sink.write(OutputKind.DNG, MediaStoreWriter.baseName(meta.captureTimeMillis), meta.captureTimeMillis) { out ->
                Channels.newChannel(out).let { channel -> while (file.hasRemaining()) channel.write(file) }
            }
            ctx.addOutput(saved)
        } finally {
            NativeMemory.free(file)
        }
    }
}
