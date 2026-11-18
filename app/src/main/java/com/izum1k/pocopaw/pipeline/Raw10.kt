package com.izum1k.pocopaw.pipeline

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Pure-Kotlin MIPI RAW10 unpacker/packer (`ImageFormat.RAW10`).
 * Every 4 consecutive 10-bit pixels `(p0..p3)` pack into 5 bytes `[b0, b1, b2, b3, lsb]`:
 * `p_i = (b_i << 2) | ((lsb ushr (2 * i)) and 0x03)`.
 * Packed payload is `ceil(width * height * 10 / 8)` bytes (62.5% of RAW16).
 */
object Raw10Unpacker {
    fun packedRowBytes(width: Int): Int {
        require(width > 0 && width % 4 == 0) { "RAW10 width must be positive and divisible by 4, got $width" }
        return (width * 10 + 7) / 8
    }

    fun packedSizeBytes(width: Int, height: Int): Long {
        require(width > 0 && height > 0 && width % 4 == 0) {
            "RAW10 dimensions must be positive and width divisible by 4, got ${width}x${height}"
        }
        return (width.toLong() * height.toLong() * 10L + 7L) / 8L
    }

    fun unpackRaw10(src: ByteBuffer, width: Int, height: Int, rowStride: Int, dst: ByteBuffer) {
        val rowBytes = packedRowBytes(width)
        require(height > 0) { "RAW10 height must be positive, got $height" }
        require(rowStride >= 0) { "RAW10 rowStride must be >= 0, got $rowStride" }
        val stride = if (rowStride > 0) rowStride else rowBytes
        require(stride >= rowBytes) { "RAW10 rowStride ($stride) < packed row bytes ($rowBytes)" }
        val minRequired = (height - 1).toLong() * stride.toLong() + rowBytes.toLong()
        require(src.remaining().toLong() >= minRequired) {
            "RAW10 input buffer truncated: remaining=${src.remaining()}, required=$minRequired"
        }
        val dstRequired = width.toLong() * height.toLong() * 2L
        require(dst.remaining().toLong() >= dstRequired) {
            "RAW16 output buffer too small: remaining=${dst.remaining()}, required=$dstRequired"
        }

        val s = src.duplicate()
        val basePos = s.position()
        val d = dst.duplicate().order(ByteOrder.nativeOrder())
        val groups = width / 4
        for (y in 0 until height) {
            var rOff = basePos + y * stride
            for (g in 0 until groups) {
                val b0 = s.get(rOff).toInt() and 0xFF
                val b1 = s.get(rOff + 1).toInt() and 0xFF
                val b2 = s.get(rOff + 2).toInt() and 0xFF
                val b3 = s.get(rOff + 3).toInt() and 0xFF
                val lsb = s.get(rOff + 4).toInt() and 0xFF
                d.putShort(((b0 shl 2) or (lsb and 0x03)).toShort())
                d.putShort(((b1 shl 2) or ((lsb ushr 2) and 0x03)).toShort())
                d.putShort(((b2 shl 2) or ((lsb ushr 4) and 0x03)).toShort())
                d.putShort(((b3 shl 2) or ((lsb ushr 6) and 0x03)).toShort())
                rOff += 5
            }
        }
    }

    fun unpackRaw10(src: ByteArray, width: Int, height: Int, rowStride: Int = 0): ShortArray {
        val rowBytes = packedRowBytes(width)
        require(height > 0) { "RAW10 height must be positive, got $height" }
        require(rowStride >= 0) { "RAW10 rowStride must be >= 0, got $rowStride" }
        val stride = if (rowStride > 0) rowStride else rowBytes
        require(stride >= rowBytes) { "RAW10 rowStride ($stride) < packed row bytes ($rowBytes)" }
        val minRequired = (height - 1).toLong() * stride.toLong() + rowBytes.toLong()
        require(src.size.toLong() >= minRequired) {
            "RAW10 input array truncated: size=${src.size}, required=$minRequired"
        }

        val out = ShortArray(width * height)
        val groups = width / 4
        var di = 0
        for (y in 0 until height) {
            var rOff = y * stride
            for (g in 0 until groups) {
                val b0 = src[rOff].toInt() and 0xFF
                val b1 = src[rOff + 1].toInt() and 0xFF
                val b2 = src[rOff + 2].toInt() and 0xFF
                val b3 = src[rOff + 3].toInt() and 0xFF
                val lsb = src[rOff + 4].toInt() and 0xFF
                out[di++] = ((b0 shl 2) or (lsb and 0x03)).toShort()
                out[di++] = ((b1 shl 2) or ((lsb ushr 2) and 0x03)).toShort()
                out[di++] = ((b2 shl 2) or ((lsb ushr 4) and 0x03)).toShort()
                out[di++] = ((b3 shl 2) or ((lsb ushr 6) and 0x03)).toShort()
                rOff += 5
            }
        }
        return out
    }

    fun packRaw10(src: ShortArray, width: Int, height: Int): ByteArray {
        val rowBytes = packedRowBytes(width)
        require(height > 0) { "RAW10 height must be positive, got $height" }
        require(src.size >= width * height) { "RAW16 input array too small: size=${src.size}, need=${width * height}" }

        val out = ByteArray(rowBytes * height)
        val groups = width / 4
        var si = 0
        for (y in 0 until height) {
            var rOff = y * rowBytes
            for (g in 0 until groups) {
                val p0 = (src[si++].toInt() and 0xFFFF).coerceAtMost(1023)
                val p1 = (src[si++].toInt() and 0xFFFF).coerceAtMost(1023)
                val p2 = (src[si++].toInt() and 0xFFFF).coerceAtMost(1023)
                val p3 = (src[si++].toInt() and 0xFFFF).coerceAtMost(1023)
                out[rOff] = (p0 ushr 2).toByte()
                out[rOff + 1] = (p1 ushr 2).toByte()
                out[rOff + 2] = (p2 ushr 2).toByte()
                out[rOff + 3] = (p3 ushr 2).toByte()
                out[rOff + 4] = ((p0 and 0x03) or ((p1 and 0x03) shl 2) or ((p2 and 0x03) shl 4) or ((p3 and 0x03) shl 6)).toByte()
                rOff += 5
            }
        }
        return out
    }
}
