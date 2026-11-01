package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CaptureResult
import android.media.ExifInterface
import android.os.Build
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

private const val TAG = "PocoPawPipeline"

/** Hands the burst to `libburstpipe`, builds the gray pyramids, aligns every frame to the reference and fits the noise model. */
class AlignStage : PipelineStage {
    override val name = "align"

    override suspend fun run(ctx: PipelineContext) {
        val native = ctx.own(NativeBurst(ctx.burst))
        ctx[PipelineKeys.Native] = native
        native.align()
    }
}

/** Robust merge of the aligned frames: spatial-domain tiles on the Bayer mosaic, or kernel regression straight to RGB. */
class MergeStage(private val kernelRegression: Boolean) : PipelineStage {
    override val name = if (kernelRegression) "merge_kr" else "merge"

    override suspend fun run(ctx: PipelineContext) {
        ctx.require(PipelineKeys.Native).merge(kernelRegression)
    }
}

/** LSC / WB, demosaic (classic merge only), CCM, chroma denoise, local tone mapping, look and sharpening; then the bitmap. */
class FinishStage(private val look: BurstLook) : PipelineStage {
    override val name = "finish"

    override suspend fun run(ctx: PipelineContext) {
        val native = ctx.require(PipelineKeys.Native)
        native.finish(look)
        val rendered = native.render()
        Log.i(TAG, native.summary())
        native.close()
        ctx[PipelineKeys.Rendered] = rendered
        ctx[PipelineKeys.QuickLookUpright] = upright(fitWithin(rendered, THUMB_LONG_SIDE), ctx.burst.metadata.orientationDegrees)
    }

    private fun fitWithin(src: Bitmap, maxSide: Int): Bitmap {
        val scale = maxSide / max(src.width, src.height).toFloat()
        if (scale >= 1f) return src
        return Bitmap.createScaledBitmap(src, (src.width * scale).roundToInt().coerceAtLeast(1), (src.height * scale).roundToInt().coerceAtLeast(1), true)
    }

    private fun upright(src: Bitmap, degrees: Int): Bitmap =
        if (degrees == 0) src else Bitmap.createBitmap(src, 0, 0, src.width, src.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)

    private companion object {
        const val THUMB_LONG_SIDE = 512
    }
}

/**
 * Encodes the rendered image with the platform's JPEG encoder (libjpeg-turbo) and tags it with
 * orientation and exposure. ExifInterface can only rewrite a seekable file, so the JPEG goes
 * through a temporary file before the sink.
 */
class JpegWriteStage(private val quality: Int = 95) : PipelineStage {
    override val name = "jpeg"

    override suspend fun run(ctx: PipelineContext) {
        val rendered = ctx.require(PipelineKeys.Rendered)
        val meta = ctx.burst.metadata
        val tmp = File.createTempFile("pocopaw", ".jpg")
        try {
            tmp.outputStream().buffered().use { rendered.compress(Bitmap.CompressFormat.JPEG, quality, it) }
            writeExif(tmp, ctx)
            val saved = ctx.sink.write(OutputKind.JPEG, MediaStoreWriter.baseName(meta.captureTimeMillis), meta.captureTimeMillis) { out ->
                tmp.inputStream().use { it.copyTo(out, BUFFER) }
            }
            ctx.addOutput(saved)
        } finally {
            tmp.delete()
            rendered.recycle()
        }
    }

    private fun writeExif(file: File, ctx: PipelineContext) {
        val meta = ctx.burst.metadata
        val result = ctx.burst.reference.result
        val exif = ExifInterface(file)
        val stamp = SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US).format(Date(meta.captureTimeMillis))
        exif.setAttribute(ExifInterface.TAG_ORIENTATION, exifOrientation(meta.orientationDegrees).toString())
        exif.setAttribute(ExifInterface.TAG_DATETIME, stamp)
        exif.setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, stamp)
        exif.setAttribute(ExifInterface.TAG_MAKE, Build.MANUFACTURER)
        exif.setAttribute(ExifInterface.TAG_MODEL, Build.MODEL)
        exif.setAttribute(ExifInterface.TAG_SOFTWARE, "PocoPaw (${ctx.burst.size} frames)")
        @Suppress("DEPRECATION")
        result.get(CaptureResult.SENSOR_SENSITIVITY)?.let { exif.setAttribute(ExifInterface.TAG_ISO_SPEED_RATINGS, it.toString()) }
        result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { exif.setAttribute(ExifInterface.TAG_EXPOSURE_TIME, (it / 1e9).toString()) }
        result.get(CaptureResult.LENS_APERTURE)?.let { exif.setAttribute(ExifInterface.TAG_F_NUMBER, it.toString()) }
        result.get(CaptureResult.LENS_FOCAL_LENGTH)?.let { exif.setAttribute(ExifInterface.TAG_FOCAL_LENGTH, "${(it * 1000).roundToInt()}/1000") }
        exif.saveAttributes()
    }

    private fun exifOrientation(degrees: Int): Int = when (degrees) {
        90 -> ExifInterface.ORIENTATION_ROTATE_90
        180 -> ExifInterface.ORIENTATION_ROTATE_180
        270 -> ExifInterface.ORIENTATION_ROTATE_270
        else -> ExifInterface.ORIENTATION_NORMAL
    }

    private companion object {
        const val BUFFER = 1 shl 16
    }
}
