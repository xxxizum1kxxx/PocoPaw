package com.izum1k.pocopaw.pipeline

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Raw10Test {

    @Test
    fun raw10UnpackBitExactAcrossAll4Phases() {
        val w = 16
        val h = 4
        val src = ShortArray(w * h) { i ->
            val phase = i % 4
            ((i * 17 + phase * 257) and 0x3FF).toShort()
        }
        val packed = Raw10Unpacker.packRaw10(src, w, h)
        val unpackedArray = Raw10Unpacker.unpackRaw10(packed, w, h)
        assertArrayEquals(src, unpackedArray)

        val srcBuf = ByteBuffer.wrap(packed)
        val dstBuf = ByteBuffer.allocate(w * h * 2).order(ByteOrder.nativeOrder())
        Raw10Unpacker.unpackRaw10(srcBuf, w, h, 0, dstBuf)
        val unpackedFromBuf = ShortArray(w * h)
        dstBuf.duplicate().order(ByteOrder.nativeOrder()).asShortBuffer().get(unpackedFromBuf)
        assertArrayEquals(src, unpackedFromBuf)
    }

    @Test
    fun raw10PackedFootprintIs625PercentOfRaw16() {
        val w = 4624
        val h = 3472
        val raw16Bytes = w.toLong() * h.toLong() * 2L
        val raw10Bytes = Raw10Unpacker.packedSizeBytes(w, h)
        assertEquals(20_068_160L, raw10Bytes)
        assertEquals(raw16Bytes * 5L / 8L, raw10Bytes)
    }

    @Test
    fun raw10HandlesPaddedRowStride() {
        val w = 8
        val h = 3
        val packedRow = Raw10Unpacker.packedRowBytes(w) // 10 bytes
        val paddedStride = 16
        val src = ShortArray(w * h) { i -> ((i * 41 + 3) and 0x3FF).toShort() }
        val packedTight = Raw10Unpacker.packRaw10(src, w, h)
        val padded = ByteArray(paddedStride * h) { 0x5A.toByte() }
        for (y in 0 until h) {
            System.arraycopy(packedTight, y * packedRow, padded, y * paddedStride, packedRow)
        }
        val unpacked = Raw10Unpacker.unpackRaw10(padded, w, h, rowStride = paddedStride)
        assertArrayEquals(src, unpacked)
    }

    @Test(expected = IllegalArgumentException::class)
    fun raw10TruncatedBufferThrowsIllegalArgument() {
        val w = 8
        val h = 4
        val truncated = ByteArray(Raw10Unpacker.packedSizeBytes(w, h).toInt() - 1)
        Raw10Unpacker.unpackRaw10(truncated, w, h)
    }

    @Test
    fun raw10AllZeroAndAll0x3ffBitPatternsHaveNoSignExtension() {
        val w = 8
        val h = 2
        val zeros = ShortArray(w * h) { 0 }
        val ones = ShortArray(w * h) { 1023 }

        val packed0 = Raw10Unpacker.packRaw10(zeros, w, h)
        assertTrueAllEqual(0, Raw10Unpacker.unpackRaw10(packed0, w, h))

        val packed1023 = Raw10Unpacker.packRaw10(ones, w, h)
        assertTrueAllEqual(1023, Raw10Unpacker.unpackRaw10(packed1023, w, h))
    }

    @Test(expected = IllegalArgumentException::class)
    fun raw10NegativeRowStrideThrowsIllegalArgument() {
        val w = 8
        val h = 2
        Raw10Unpacker.unpackRaw10(ByteArray(32), w, h, rowStride = -16)
    }

    private fun assertTrueAllEqual(expected: Int, actual: ShortArray) {
        for (s in actual) {
            assertEquals(expected, s.toInt() and 0xFFFF)
        }
    }
}
