package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import android.media.ExifInterface
import android.os.Build
import java.nio.ByteBuffer
import kotlin.math.max

/**
 * The merged burst as DNG. The native session writes the mosaic (`dng.h`: 16 bits keeping 4 of the merge's extra bits,
 * lens shading as GainMap opcodes, bracket headroom as BaselineExposure); this gathers what Camera2 knows about the
 * camera: the colour and forward matrix per calibration illuminant, the reference frame's neutral, orientation and the
 * quick look as thumbnail. Null when the session has no mosaic (kernel regression) or the camera reports no colour matrix.
 */
class MergedDngWriter(private val metadata: RawMetadata, private val reference: CaptureResult) {

    fun encode(native: NativeBurst, thumbnail: Bitmap?): ByteBuffer? {
        val chars = metadata.characteristics
        val color = ArrayList<Float>(36)
        val illuminants = ArrayList<Int>(2)
        val calibrations = listOf(
            Triple(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1, CameraCharacteristics.SENSOR_FORWARD_MATRIX1, chars.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1)),
            Triple(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2, CameraCharacteristics.SENSOR_FORWARD_MATRIX2, chars.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2)?.toInt()),
        )
        for ((cm, fm, illuminant) in calibrations) {
            val c = chars.get(cm) ?: break
            val f = chars.get(fm) ?: break
            color += c.rowMajor()
            color += f.rowMajor()
            illuminants += illuminant ?: break
        }
        if (illuminants.isEmpty()) return null
        val thumb = thumbnail?.let { fitWithin(it, THUMB_MAX) }
        return native.encodeDng(
            color.toFloatArray(),
            illuminants.toIntArray(),
            neutral(),
            orientation(metadata.orientationDegrees),
            Build.MANUFACTURER,
            Build.MODEL,
            "PocoPaw",
            thumb?.let(::rgb),
            thumb?.width ?: 0,
            thumb?.height ?: 0,
        )
    }

    private fun neutral(): FloatArray {
        val point = reference.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT) ?: metadata.neutralPoint
        if (point != null && point.size == 3) return FloatArray(3) { point[it].toFloat() }
        val g = metadata.wbGains ?: return floatArrayOf(1f, 1f, 1f)
        val green = (g.greenEven + g.greenOdd) / 2f
        return floatArrayOf(green / g.red, 1f, green / g.blue)
    }

    private fun ColorSpaceTransform.rowMajor(): List<Float> = List(9) { getElement(it % 3, it / 3).toFloat() }

    private fun rgb(bitmap: Bitmap): ByteArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val out = ByteArray(pixels.size * 3)
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i * 3] = (p shr 16).toByte()
            out[i * 3 + 1] = (p shr 8).toByte()
            out[i * 3 + 2] = p.toByte()
        }
        return out
    }

    private fun fitWithin(src: Bitmap, maxSide: Int): Bitmap {
        val scale = maxSide / max(src.width, src.height).toFloat()
        if (scale >= 1f) return src
        return Bitmap.createScaledBitmap(src, (src.width * scale).toInt().coerceAtLeast(1), (src.height * scale).toInt().coerceAtLeast(1), true)
    }

    private fun orientation(degrees: Int): Int = when (degrees) {
        90 -> ExifInterface.ORIENTATION_ROTATE_90
        180 -> ExifInterface.ORIENTATION_ROTATE_180
        270 -> ExifInterface.ORIENTATION_ROTATE_270
        else -> ExifInterface.ORIENTATION_NORMAL
    }

    private companion object {
        const val THUMB_MAX = 256
    }
}
