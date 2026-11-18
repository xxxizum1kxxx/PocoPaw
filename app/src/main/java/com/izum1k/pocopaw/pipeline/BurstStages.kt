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
import kotlin.math.roundToInt

private const val TAG = "PocoPawPipeline"

/** Picks the frame the burst is merged onto ([NativeBurst.referenceFor]), before the DNG and quick look use it. */
class ReferenceStage : PipelineStage {
    override val name = "reference"

    override suspend fun run(ctx: PipelineContext) {
        val stream = ctx.burst.stream
        ctx.burst.referenceIndex = if (stream != null) stream.reference() else NativeBurst.referenceFor(ctx.burst)
    }
}

/**
 * Hands the burst to `libburstpipe`, crops it to the digital zoom, builds the gray pyramids, aligns
 * every frame to the reference and fits the noise model (or takes the camera's calibrated [noise]). With [gpu] this and the later native steps
 * run on the GPU where they can. A streamed burst ([RawBurst.stream]) is aligned and merged already; this waits for its last frames.
 */
class AlignStage(
    private val gpu: Boolean = false,
    private val noise: NoiseCurve? = null,
    private val accelBackend: AccelBackend = AccelBackend.AUTO,
    private val flowMethod: FlowMethod = FlowMethod.BLOCK_MATCH,
    private val flowModelPath: String = "",
) : PipelineStage {
    override val name = "align"

    override suspend fun run(ctx: PipelineContext) {
        ctx.burst.stream?.let { stream ->
            ctx[PipelineKeys.Native] = ctx.own(stream.finish())
            return
        }
        val native = ctx.own(NativeBurst(ctx.burst))
        ctx[PipelineKeys.Native] = native
        native.useGpu(gpu)
        if (accelBackend != AccelBackend.AUTO) native.setAccelBackend(accelBackend)
        if (flowMethod != FlowMethod.BLOCK_MATCH) native.setFlowOptions(flowMethod, flowModelPath)
        noise?.let { native.setNoiseCurve(it) }
        val zoom = ctx.burst.metadata.cropZoom
        if (zoom > 1f) native.crop(zoom)
        native.align()
    }
}

/**
 * Robust merge of the aligned frames: spatial- or frequency-domain tiles on the Bayer mosaic, or
 * kernel regression straight to RGB.
 */
class MergeStage(private val options: MergeOptions) : PipelineStage {
    override val name = if (options.kernelRegression) "merge_kr" else "merge"

    override suspend fun run(ctx: PipelineContext) {
        ctx.require(PipelineKeys.Native).merge(options)
    }
}

/**
 * LSC / WB, demosaic (classic merge only), CCM, chroma denoise, local tone mapping, look and
 * sharpening; renders the small viewfinder thumbnail and encodes the JPEG / Ultra HDR directly in
 * native memory without allocating a full-resolution Java Bitmap.
 */
class FinishStage(
    private val look: BurstLook,
    private val ultraHdr: Boolean = false,
    private val aspectRatio: Float = 0f,
    private val writeJpeg: Boolean = true,
    private val quality: Int = 95,
) : PipelineStage {
    override val name = "finish"

    override suspend fun run(ctx: PipelineContext) {
        val native = ctx.require(PipelineKeys.Native)
        if (aspectRatio > 0f) native.setAspect(aspectRatio)
        val meta = ctx.burst.metadata
        native.finish(
            look,
            keepHdr = ultraHdr && writeJpeg,
            faceBoxes = meta.faceBoxes,
            orientationDegrees = meta.orientationDegrees,
        )
        val thumb = native.renderThumbnail(THUMB_LONG_SIDE)
        val uprightThumb = upright(thumb, ctx.burst.metadata.orientationDegrees)
        if (uprightThumb !== thumb) thumb.recycle()
        ctx[PipelineKeys.QuickLookUpright] = uprightThumb
        if (writeJpeg) {
            native.encodePhoto(quality = quality, wantUltraHdr = ultraHdr)?.let { ctx[PipelineKeys.EncodedJpeg] = it }
        }
        Log.i(TAG, native.summary())
        native.close()
    }

    private fun upright(src: Bitmap, degrees: Int): Bitmap =
        if (degrees == 0) src else Bitmap.createBitmap(src, 0, 0, src.width, src.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)

    private companion object {
        const val THUMB_LONG_SIDE = 512
    }
}

/**
 * Tags the natively encoded JPEG (or Ultra HDR JPEG) with orientation and exposure EXIF and writes
 * it to the sink. ExifInterface only rewrites a seekable file, so the encoded bytes pass through a
 * temporary file; it rewrites only the primary image's EXIF segment and keeps any gain map intact.
 */
class JpegWriteStage : PipelineStage {
    override val name = "jpeg"

    override suspend fun run(ctx: PipelineContext) {
        val encoded = ctx.require(PipelineKeys.EncodedJpeg)
        val meta = ctx.burst.metadata
        val tmp = File.createTempFile("pocopaw", ".jpg")
        try {
            tmp.writeBytes(encoded.bytes)
            writeExif(tmp, ctx)
            val kind = if (encoded.ultraHdr) OutputKind.ULTRA_HDR else OutputKind.JPEG
            val saved = ctx.sink.write(kind, MediaStoreWriter.baseName(meta.captureTimeMillis), meta.captureTimeMillis) { out ->
                tmp.inputStream().use { it.copyTo(out, BUFFER) }
            }
            ctx.addOutput(saved)
        } finally {
            tmp.delete()
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
        meta.cropZoom.takeIf { it > 1f }?.let { exif.setAttribute(ExifInterface.TAG_DIGITAL_ZOOM_RATIO, it.toString()) }
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
