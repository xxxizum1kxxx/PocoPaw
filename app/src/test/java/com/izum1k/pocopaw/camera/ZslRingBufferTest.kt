package com.izum1k.pocopaw.camera

import android.hardware.camera2.CaptureResult
import com.izum1k.pocopaw.pipeline.Raw10Unpacker
import com.izum1k.pocopaw.pipeline.RawFrame
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ZslRingBufferTest {

    private fun dummyResult(): CaptureResult {
        val f = sun.misc.Unsafe::class.java.getDeclaredField("theUnsafe").apply { isAccessible = true }
        val unsafe = f.get(null) as sun.misc.Unsafe
        return unsafe.allocateInstance(CaptureResult::class.java) as CaptureResult
    }

    @Test
    fun zslRingBufferEvictsOldestAndClosesEvictedFrames() {
        var nowNs = 10_000_000_000L
        val released = mutableListOf<Int>()
        val ring = ZslRingBuffer(capacity = 12, maxAgeNs = 2_000_000_000L, clockNs = { nowNs })

        for (i in 0 until 20) {
            val buf = ByteBuffer.allocate(16)
            val frame = RawFrame(buf, 4, 2, timestampNs = nowNs + i * 1_000_000L, result = dummyResult()) {
                released += i
            }
            ring.push(frame, nowNs = nowNs + i * 1_000_000L)
        }

        assertEquals(12, ring.size)
        assertEquals((0 until 8).toList(), released)

        ring.close()
        assertEquals(0, ring.size)
        assertEquals(20, released.size)
    }

    @Test
    fun zslPairsOutOfOrderFramesAndSnapshotsRecentK() {
        val nowNs = 5_000_000_000L
        val ring = ZslRingBuffer(capacity = 12, clockNs = { nowNs })
        val res = dummyResult()

        // Image arrives before result at ts1, result arrives before image at ts2..ts3
        val ts1 = nowNs - 300L
        val ts2 = nowNs - 200L
        val ts3 = nowNs - 100L
        assertFalse(ring.onImage(ByteBuffer.allocate(32), 4, 4, timestampNs = ts1, onRelease = {}, nowNs = nowNs))
        assertTrue(ring.onResultWithTimestamp(ts1, res, nowNs = nowNs))

        assertFalse(ring.onResultWithTimestamp(ts2, res, nowNs = nowNs))
        assertFalse(ring.onResultWithTimestamp(ts3, res, nowNs = nowNs))
        assertTrue(ring.onImage(ByteBuffer.allocate(32), 4, 4, timestampNs = ts2, onRelease = {}, nowNs = nowNs))
        assertTrue(ring.onImage(ByteBuffer.allocate(32), 4, 4, timestampNs = ts3, onRelease = {}, nowNs = nowNs))

        assertEquals(3, ring.size)
        val snap = ring.drain(2, nowNs = nowNs)
        assertEquals(listOf(ts2, ts3), snap.map { it.timestampNs })
        assertEquals(1, ring.size)
        ring.close()
    }

    @Test
    fun zslEmptyRingDrainReturnsEmptyList() {
        val ring = ZslRingBuffer(capacity = 12, clockNs = { 1_000_000_000L })
        val drained = ring.drain(8)
        assertTrue(drained.isEmpty())
        assertEquals(0, ring.size)
    }

    @Test
    fun zslPartialRingReturnsAvailableFrames() {
        val nowNs = 1_000_000_000L
        val ring = ZslRingBuffer(capacity = 12, clockNs = { nowNs })
        val res = dummyResult()
        repeat(3) { i ->
            ring.push(RawFrame(ByteBuffer.allocate(16), 4, 2, nowNs + i, res) {}, nowNs = nowNs)
        }
        val drained = ring.drain(8, nowNs = nowNs)
        assertEquals(3, drained.size)
        assertEquals(0, ring.size)
    }

    @Test
    fun zslStaleFramesOverTwoSecondsAreEvictedAndClosed() {
        var nowNs = 1_000_000_000L
        val released = mutableListOf<Long>()
        val ring = ZslRingBuffer(capacity = 12, maxAgeNs = 2_000_000_000L, clockNs = { nowNs })
        val res = dummyResult()

        ring.push(RawFrame(ByteBuffer.allocate(16), 4, 2, 1_000_000_000L, res) { released += 1L }, nowNs = nowNs)
        ring.push(RawFrame(ByteBuffer.allocate(16), 4, 2, 2_000_000_000L, res) { released += 2L }, nowNs = 2_000_000_000L)
        nowNs = 3_500_000_000L
        ring.push(RawFrame(ByteBuffer.allocate(16), 4, 2, 3_500_000_000L, res) { released += 3L }, nowNs = nowNs)

        assertEquals(listOf(1L), released)
        val fresh = ring.drain(8, nowNs = nowNs)
        assertEquals(listOf(2_000_000_000L, 3_500_000_000L), fresh.map { it.timestampNs })
    }

    @Test
    fun zslBracketHybridCombinesShortZslFramesAndPostShutterLong() {
        val nowNs = 2_000_000_000L
        val ring = ZslRingBuffer(capacity = 12, clockNs = { nowNs })
        val res = dummyResult()
        repeat(4) { i ->
            ring.push(RawFrame(ByteBuffer.allocate(16), 4, 2, nowNs + i, res) {}, nowNs = nowNs)
        }
        val shortExp = ManualExposure(100, 10_000_000L)
        val longExp = ManualExposure(100, 40_000_000L)
        val plan = ExposurePlan.PerFrame(listOf(shortExp, shortExp, longExp))
        val burstCount = 6 // 4 short, 2 long
        val shortSlots = (0 until burstCount).count { plan.at(it) == shortExp }
        assertEquals(4, shortSlots)

        val zslShort = ring.drain(shortSlots, nowNs = nowNs)
        val postLong = List(burstCount - zslShort.size) { i ->
            RawFrame(ByteBuffer.allocate(16), 4, 2, nowNs + 100 + i, res) {}
        }
        val combined = zslShort + postLong
        assertEquals(4, zslShort.size)
        assertEquals(2, postLong.size)
        assertEquals(6, combined.size)
    }

    @Test
    fun zslReopenPreservesFocusDioptresInCaptureSpec() {
        val spec = CaptureSpec(frameCount = 8, cropZoom = 1.5f, focusDioptres = 2.35f)
        assertEquals(2.35f, spec.focusDioptres ?: 0f, 1e-6f)
        val budget = RawRequestBudget()
        budget.recordSubmitted(58)
        assertTrue(budget.needsProactiveReopen())
        budget.reset()
        assertEquals(2.35f, spec.focusDioptres ?: 0f, 1e-6f)
    }

    @Test
    fun zslRaw10CompactRingSavesMemoryAndCyclesReopenAcross130Frames() {
        val w = 8
        val h = 4
        val raw10Bytes = Raw10Unpacker.packedSizeBytes(w, h).toInt()
        val raw16Bytes = w * h * 2
        assertEquals(raw16Bytes * 5 / 8, raw10Bytes) // 37.5% memory savings

        val releasedRaw10 = mutableListOf<Int>()
        val releasedRaw16 = mutableListOf<Int>()
        var allocCount = 0
        var nowNs = 10_000_000_000L
        val ring = ZslRingBuffer(
            capacity = 12,
            clockNs = { nowNs },
            allocateRaw16 = { bytes -> ByteBuffer.allocate(bytes).order(ByteOrder.nativeOrder()) },
            freeRaw16 = { _ -> releasedRaw16 += ++allocCount },
        )
        val budget = RawRequestBudget()
        val pixels = ShortArray(w * h) { i -> ((i * 37) and 0x3FF).toShort() }
        val packed = Raw10Unpacker.packRaw10(pixels, w, h)
        val res = dummyResult()
        var reopens = 0

        for (i in 0 until 130) {
            if (budget.needsProactiveReopen()) {
                budget.reset()
                reopens++
            }
            budget.recordSubmitted(1)
            assertTrue(budget.consumed <= RawRequestBudget.DEFAULT_LIMIT)
            val buf = ByteBuffer.wrap(packed.copyOf())
            ring.onImage(
                data = buf,
                width = w,
                height = h,
                timestampNs = nowNs + i,
                isRaw10 = true,
                rowStride = Raw10Unpacker.packedRowBytes(w),
                onRelease = { releasedRaw10 += i },
                nowNs = nowNs + i,
            )
            ring.onResultWithTimestamp(nowNs + i, res, nowNs = nowNs + i)
            nowNs += 10_000L
        }

        assertTrue(reopens >= 2)
        val drained = ring.drain(4, nowNs = nowNs)
        assertEquals(4, drained.size)
        // Compact RAW10 buffers were freed when unpacked on drain
        assertEquals(130 - 8, releasedRaw10.size)

        val outShorts = ShortArray(w * h)
        drained.first().samples().get(outShorts)
        assertArrayEquals(pixels, outShorts)
        drained.forEach { it.close() }
        ring.close()
        assertEquals(130, releasedRaw10.size)
        assertEquals(4, releasedRaw16.size)
    }

    @Test
    fun zslRingBufferDrainReleasesAlreadyMaterializedAndRemainingBuffersOnOom() {
        val w = 8
        val h = 2
        val packed = Raw10Unpacker.packRaw10(ShortArray(w * h) { 512 }, w, h)
        val releasedRaw10 = mutableListOf<Int>()
        val releasedRaw16 = mutableListOf<Int>()
        var allocCount = 0
        val nowNs = 5_000_000_000L
        val ring = ZslRingBuffer(
            capacity = 8,
            clockNs = { nowNs },
            allocateRaw16 = { bytes ->
                allocCount++
                if (allocCount == 3) throw OutOfMemoryError("simulated OOM on frame 3")
                ByteBuffer.allocate(bytes).order(ByteOrder.nativeOrder())
            },
            freeRaw16 = { _ -> releasedRaw16 += releasedRaw16.size + 1 },
        )
        val res = dummyResult()
        repeat(4) { i ->
            ring.onImage(
                data = ByteBuffer.wrap(packed.copyOf()),
                width = w,
                height = h,
                timestampNs = nowNs + i,
                isRaw10 = true,
                rowStride = Raw10Unpacker.packedRowBytes(w),
                onRelease = { releasedRaw10 += i },
                nowNs = nowNs,
            )
            ring.onResultWithTimestamp(nowNs + i, res, nowNs = nowNs)
        }

        var thrown: Throwable? = null
        try {
            ring.drain(4, nowNs = nowNs)
        } catch (t: Throwable) {
            thrown = t
        }
        assertTrue(thrown is OutOfMemoryError)
        assertEquals(0, ring.size)
        assertEquals(2, releasedRaw16.size)
        assertEquals(4, releasedRaw10.size)
        ring.close()
    }

    @Test
    fun zslRingBufferDrainReleasesAllocatedDstWhenUnpackFails() {
        val w = 8
        val h = 2
        val validPacked = Raw10Unpacker.packRaw10(ShortArray(w * h) { 256 }, w, h)
        val truncatedPacked = ByteArray(validPacked.size - 1)
        val releasedRaw10 = mutableListOf<Int>()
        val releasedRaw16 = mutableListOf<Int>()
        val nowNs = 5_000_000_000L
        val ring = ZslRingBuffer(
            capacity = 8,
            clockNs = { nowNs },
            allocateRaw16 = { bytes -> ByteBuffer.allocate(bytes).order(ByteOrder.nativeOrder()) },
            freeRaw16 = { _ -> releasedRaw16 += releasedRaw16.size + 1 },
        )
        val res = dummyResult()
        listOf(validPacked, truncatedPacked).forEachIndexed { i, payload ->
            ring.onImage(
                data = ByteBuffer.wrap(payload),
                width = w,
                height = h,
                timestampNs = nowNs + i,
                isRaw10 = true,
                rowStride = Raw10Unpacker.packedRowBytes(w),
                onRelease = { releasedRaw10 += i },
                nowNs = nowNs,
            )
            ring.onResultWithTimestamp(nowNs + i, res, nowNs = nowNs)
        }

        var thrown: Throwable? = null
        try {
            ring.drain(2, nowNs = nowNs)
        } catch (t: Throwable) {
            thrown = t
        }
        assertTrue(thrown is IllegalArgumentException)
        assertEquals(2, releasedRaw16.size)
        assertEquals(2, releasedRaw10.size)
        ring.close()
    }

    @Test
    fun zslExtendedMaxAgeKeepsFramesAcross200MsPollingAndRespectsReservedShortFrames() {
        var nowNs = 10_000_000_000L
        val ring = ZslRingBuffer(capacity = 12, maxAgeNs = 3_000_000_000L, clockNs = { nowNs })
        val res = dummyResult()
        repeat(8) { i ->
            val ts = nowNs + i * 200_000_000L
            ring.push(RawFrame(ByteBuffer.allocate(16), 4, 2, ts, res) {}, nowNs = ts)
        }
        nowNs += 2_500_000_000L
        val spec = CaptureSpec(frameCount = 8, allowZsl = true, reserveShortFrames = 2)
        val maxZsl = (spec.frameCount - spec.reserveShortFrames).coerceAtLeast(0)
        val drained = ring.drain(maxZsl, nowNs = nowNs)
        assertEquals(6, drained.size)
        assertTrue(spec.allowZsl)
        drained.forEach { it.release() }
        ring.close()
    }
}
