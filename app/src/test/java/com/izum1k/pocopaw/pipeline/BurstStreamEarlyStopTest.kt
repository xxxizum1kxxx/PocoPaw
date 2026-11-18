package com.izum1k.pocopaw.pipeline

import android.hardware.camera2.CaptureResult
import com.izum1k.pocopaw.camera.RawRequestBudget
import com.izum1k.pocopaw.camera.ZslRingBuffer
import com.izum1k.pocopaw.data.CameraSettings
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.log10

class BurstStreamEarlyStopTest {

    private class FakeStreamSession(
        private val perFrameWeight: Float = 0.9f,
        private val refToPick: Int = 1,
    ) : StreamSession {
        var opts: MergeOptions = MergeOptions()
        var added = 0
        var ended = false
        var neff = 1.0f

        override fun useGpu(enabled: Boolean) = Unit
        override fun setAccelBackend(backend: AccelBackend) = Unit
        override fun setNoiseCurve(curve: NoiseCurve) = Unit
        override fun crop(zoom: Float) = Unit
        override fun beginStream(options: MergeOptions) {
            opts = options
        }

        override fun addFrameWithProgress(frame: RawFrame): StreamProgress {
            added++
            if (added < 3) {
                return StreamProgress(-1, added, 1.0f, 0f, 0f, false)
            }
            if (added == 3) {
                neff = (1.0f + 2f * perFrameWeight).coerceIn(1.0f, 3.0f)
            } else {
                neff = (neff + perFrameWeight).coerceIn(1.0f, added.toFloat())
            }
            val snrDb = 10.0f * log10(neff.coerceAtLeast(1.0f))
            val stop = opts.earlyStopEnabled && snrDb >= opts.earlyStopTargetSnrDb
            return StreamProgress(refToPick, added, neff, 0.5f, snrDb, stop)
        }

        override fun endFrames(): Int {
            ended = true
            return if (added > 0) minOf(refToPick, added - 1) else 0
        }

        override fun close() = Unit
    }

    private fun dummyMeta(): RawMetadata {
        val f = sun.misc.Unsafe::class.java.getDeclaredField("theUnsafe").apply { isAccessible = true }
        val unsafe = f.get(null) as sun.misc.Unsafe
        return unsafe.allocateInstance(RawMetadata::class.java) as RawMetadata
    }

    private fun dummyResult(): CaptureResult {
        val f = sun.misc.Unsafe::class.java.getDeclaredField("theUnsafe").apply { isAccessible = true }
        val unsafe = f.get(null) as sun.misc.Unsafe
        return unsafe.allocateInstance(CaptureResult::class.java) as CaptureResult
    }

    @Test
    fun earlyStopStopsAfterMinimumCandidatesAndSyncsRawRequestBudget() = runBlocking {
        val fake = FakeStreamSession(perFrameWeight = 0.95f)
        val released = mutableListOf<Int>()
        val budget = RawRequestBudget()
        val meta = dummyMeta()
        val res = dummyResult()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(earlyStopEnabled = true, earlyStopTargetSnrDb = 5.0f),
            sessionFactory = { _, _ -> fake },
        )

        var submitted = 0
        val chunkSize = 4
        while (submitted < 16) {
            val chunk = minOf(chunkSize, 16 - submitted)
            budget.consume(chunk)
            repeat(chunk) {
                val idx = submitted++
                stream.add(RawFrame(ByteBuffer.allocate(8), 2, 2, idx.toLong(), res) { released += idx }, meta)
            }
            val sp = stream.awaitProgress()
            if (sp?.shouldStop == true || stream.shouldStop) break
        }

        assertTrue(stream.shouldStop)
        assertEquals(4, submitted)
        assertEquals(65 - 4, budget.remaining.value)
        assertEquals(1, stream.reference())
        stream.finishSession()
        assertTrue(fake.ended)
        stream.close()
    }

    @Test
    fun neffDoesNotInflateWhenBlurryFramesAreRejected() = runBlocking {
        val fake = FakeStreamSession(perFrameWeight = 0.0f)
        val meta = dummyMeta()
        val res = dummyResult()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(earlyStopEnabled = true, earlyStopTargetSnrDb = 4.0f),
            sessionFactory = { _, _ -> fake },
        )
        repeat(6) { i ->
            stream.add(RawFrame(ByteBuffer.allocate(8), 2, 2, i.toLong(), res) {}, meta)
        }
        val sp = stream.awaitProgress()!!
        assertEquals(1.0f, sp.neff, 1e-5f)
        assertFalse(sp.shouldStop)
        stream.finishSession()
        stream.close()
    }

    @Test
    fun cancelStreamAtTwoFramesBeforeCandidatesFinalizesSafely() = runBlocking {
        val fake = FakeStreamSession(perFrameWeight = 0.9f)
        val meta = dummyMeta()
        val res = dummyResult()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(earlyStopEnabled = true, earlyStopTargetSnrDb = 4.0f),
            sessionFactory = { _, _ -> fake },
        )
        repeat(2) { i ->
            stream.add(RawFrame(ByteBuffer.allocate(8), 2, 2, i.toLong(), res) {}, meta)
        }
        val sp = stream.awaitProgress()!!
        assertFalse(sp.shouldStop)
        val ref = stream.reference()
        assertEquals(1, ref)
        stream.finishSession()
        assertTrue(fake.ended)
        stream.close()
    }

    @Test
    fun earlyStopWithTwoXZoomBoostAllocatesAndSavesBudget() = runBlocking {
        val boostedFrames = BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 2.0f)
        assertEquals(16, boostedFrames)

        val budget = RawRequestBudget()
        val fake = FakeStreamSession(perFrameWeight = 0.95f)
        val meta = dummyMeta()
        val res = dummyResult()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(earlyStopEnabled = true, earlyStopTargetSnrDb = 5.5f),
            sessionFactory = { _, _ -> fake },
        )
        var submitted = 0
        while (submitted < boostedFrames) {
            val chunk = minOf(4, boostedFrames - submitted)
            budget.consume(chunk)
            repeat(chunk) {
                stream.add(RawFrame(ByteBuffer.allocate(8), 2, 2, submitted++.toLong(), res) {}, meta)
            }
            if (stream.awaitProgress()?.shouldStop == true) break
        }
        assertTrue(submitted < boostedFrames)
        assertEquals(65 - submitted, budget.remaining.value)
        stream.finishSession()
        stream.close()
    }

    @Test
    fun earlyStopWithZslRingConsumesBufferedFramesFirst() = runBlocking {
        val nowNs = 1_000_000_000L
        val ring = ZslRingBuffer(capacity = 12, clockNs = { nowNs })
        val res = dummyResult()
        repeat(4) { i ->
            ring.push(RawFrame(ByteBuffer.allocate(8), 2, 2, nowNs + i, res) {}, nowNs = nowNs)
        }

        val fake = FakeStreamSession(perFrameWeight = 0.95f)
        val meta = dummyMeta()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(earlyStopEnabled = true, earlyStopTargetSnrDb = 5.0f),
            sessionFactory = { _, _ -> fake },
        )

        val zslFrames = ring.drain(8, nowNs = nowNs)
        assertEquals(4, zslFrames.size)
        zslFrames.forEach { stream.add(it, meta) }
        val sp = stream.awaitProgress()!!
        assertTrue(sp.shouldStop)
        assertEquals(4, sp.acceptedCount)
        stream.finishSession()
        stream.close()
    }

    @Test
    fun m4SettingsDefaultsAnd15FloatMergeOptionsContractPreserved() {
        val s = CameraSettings()
        assertFalse(s.zslEnabled)
        assertFalse(s.earlyStopEnabled)
        assertFalse(s.raw10Enabled)
        assertEquals(15, MergeOptions().toArray().size)
    }

    @Test
    fun addAfterCloseReleasesFrameWithoutRejectedExecutionException() {
        val fake = FakeStreamSession()
        val meta = dummyMeta()
        val res = dummyResult()
        val released = mutableListOf<Int>()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(),
            sessionFactory = { _, _ -> fake },
        )
        stream.close()
        stream.add(RawFrame(ByteBuffer.allocate(8), 2, 2, 1L, res) { released += 0 }, meta)
        assertEquals(listOf(0), released)
    }

    @Test
    fun workerErrorPropagatesToAwaitProgressAndFinishSessionWithCleanup() = runBlocking {
        val released = mutableListOf<Int>()
        var closedSession = false
        val failingSession = object : StreamSession {
            private var added = 0
            override fun useGpu(enabled: Boolean) = Unit
            override fun setAccelBackend(backend: AccelBackend) = Unit
            override fun setNoiseCurve(curve: NoiseCurve) = Unit
            override fun crop(zoom: Float) = Unit
            override fun beginStream(options: MergeOptions) = Unit
            override fun addFrameWithProgress(frame: RawFrame): StreamProgress {
                added++
                if (added == 2) throw OutOfMemoryError("simulated native OOM")
                return StreamProgress(-1, added, 1.0f, 0f, 0f, false)
            }
            override fun endFrames(): Int = 0
            override fun close() {
                closedSession = true
            }
        }
        val meta = dummyMeta()
        val res = dummyResult()
        val stream = BurstStream(
            gpu = false,
            noise = null,
            options = MergeOptions(earlyStopEnabled = true),
            sessionFactory = { _, _ -> failingSession },
        )

        repeat(3) { i ->
            stream.add(RawFrame(ByteBuffer.allocate(8), 2, 2, i.toLong(), res) { released += i }, meta)
        }

        var awaitThrown: Throwable? = null
        try {
            stream.awaitProgress()
        } catch (t: Throwable) {
            awaitThrown = t
        }
        assertTrue(awaitThrown is OutOfMemoryError)

        var finishThrown: Throwable? = null
        try {
            stream.finishSession()
        } catch (t: Throwable) {
            finishThrown = t
        }
        assertTrue(finishThrown is OutOfMemoryError)
        assertTrue(closedSession)
        assertEquals(listOf(0, 1, 2), released.sorted())
        stream.close()
    }
}
