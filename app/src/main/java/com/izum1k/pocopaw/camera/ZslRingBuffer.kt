package com.izum1k.pocopaw.camera

import android.hardware.camera2.CaptureResult
import android.os.SystemClock
import com.izum1k.pocopaw.pipeline.NativeMemory
import com.izum1k.pocopaw.pipeline.Raw10Unpacker
import com.izum1k.pocopaw.pipeline.RawFrame
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * Bounded native-memory ring buffer for Zero Shutter Lag (ZSL) RAW frames.
 * Pairs arriving RAW buffers (`RAW_SENSOR` or compact `RAW10`) with [CaptureResult] via [TimestampPairs]
 * using `onDropA` cleanup, evicts stale frames older than [maxAgeNs], and immediately frees any
 * evicted or cleared native buffers. Compact `RAW10` entries sit in the ring at 62.5% size and
 * unpack to `RAW16` on [drain].
 */
class ZslRingBuffer(
    val capacity: Int = DEFAULT_CAPACITY,
    val maxAgeNs: Long = DEFAULT_MAX_AGE_NS,
    private val clockNs: () -> Long = { SystemClock.elapsedRealtimeNanos() },
    private val allocateRaw16: (Int) -> ByteBuffer = { NativeMemory.allocate(it) },
    private val freeRaw16: (ByteBuffer) -> Unit = { NativeMemory.free(it) },
) : Closeable {
    init {
        require(capacity >= 1) { "ZslRingBuffer capacity must be >= 1, got $capacity" }
        require(maxAgeNs > 0L) { "ZslRingBuffer maxAgeNs must be > 0, got $maxAgeNs" }
    }

    private data class RawEntry(
        val data: ByteBuffer,
        val width: Int,
        val height: Int,
        val timestampNs: Long,
        val isRaw10: Boolean,
        val rowStride: Int,
        val onRelease: (ByteBuffer) -> Unit,
    ) {
        fun release() = onRelease(data)
    }

    private sealed interface RingItem {
        val timestampNs: Long
        fun release()
        fun materialize(allocateRaw16: (Int) -> ByteBuffer, freeRaw16: (ByteBuffer) -> Unit): RawFrame

        class PairedFrame(val frame: RawFrame) : RingItem {
            override val timestampNs: Long get() = frame.timestampNs
            override fun release() = frame.release()
            override fun materialize(allocateRaw16: (Int) -> ByteBuffer, freeRaw16: (ByteBuffer) -> Unit): RawFrame = frame
        }

        class CompactRaw10(
            val entry: RawEntry,
            val result: CaptureResult,
        ) : RingItem {
            override val timestampNs: Long get() = entry.timestampNs
            override fun release() = entry.release()
            override fun materialize(allocateRaw16: (Int) -> ByteBuffer, freeRaw16: (ByteBuffer) -> Unit): RawFrame {
                var dst: ByteBuffer? = null
                try {
                    dst = allocateRaw16(entry.width * entry.height * 2)
                    Raw10Unpacker.unpackRaw10(entry.data, entry.width, entry.height, entry.rowStride, dst)
                    dst.rewind()
                    val frame = RawFrame(dst, entry.width, entry.height, entry.timestampNs, result, freeRaw16)
                    dst = null
                    return frame
                } catch (t: Throwable) {
                    dst?.let(freeRaw16)
                    throw t
                } finally {
                    entry.release()
                }
            }
        }
    }

    private val pairs = TimestampPairs<RawEntry, CaptureResult>(
        max = capacity * 2,
        onDropA = { it.release() },
    )
    private val ring = ArrayDeque<RingItem>(capacity)

    @get:Synchronized
    val size: Int
        get() = ring.size

    @get:Synchronized
    val unpairedImagesCount: Int
        get() = pairs.unpairedACount()

    @get:Synchronized
    val unpairedResultsCount: Int
        get() = pairs.unpairedBCount()

    @Synchronized
    fun onImage(
        data: ByteBuffer,
        width: Int,
        height: Int,
        timestampNs: Long,
        isRaw10: Boolean = false,
        rowStride: Int = 0,
        onRelease: (ByteBuffer) -> Unit = { NativeMemory.free(it) },
        nowNs: Long = clockNs(),
    ): Boolean {
        val entry = RawEntry(data, width, height, timestampNs, isRaw10, rowStride, onRelease)
        val paired = pairs.image(timestampNs, entry) ?: return false
        pushItemLocked(toRingItem(paired.first, paired.second), nowNs)
        return true
    }

    @Synchronized
    fun onResult(result: CaptureResult, nowNs: Long = clockNs()): Boolean {
        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return false
        val paired = pairs.result(ts, result) ?: return false
        pushItemLocked(toRingItem(paired.first, paired.second), nowNs)
        return true
    }

    @Synchronized
    fun onResultWithTimestamp(timestampNs: Long, result: CaptureResult, nowNs: Long = clockNs()): Boolean {
        val paired = pairs.result(timestampNs, result) ?: return false
        pushItemLocked(toRingItem(paired.first, paired.second), nowNs)
        return true
    }

    @Synchronized
    fun push(frame: RawFrame, nowNs: Long = clockNs()) {
        pushItemLocked(RingItem.PairedFrame(frame), nowNs)
    }

    @Synchronized
    fun evictStale(nowNs: Long = clockNs()): Int {
        var evicted = 0
        val it = ring.iterator()
        while (it.hasNext()) {
            val item = it.next()
            if (nowNs - item.timestampNs > maxAgeNs) {
                it.remove()
                item.release()
                evicted++
            }
        }
        return evicted
    }

    /**
     * Evicts stale frames and drains up to [maxCount] newest frames in chronological ascending order,
     * transferring buffer ownership to the caller (unpacking compact `RAW10` entries to `RAW16`).
     */
    @Synchronized
    fun drain(maxCount: Int, nowNs: Long = clockNs()): List<RawFrame> {
        evictStale(nowNs)
        if (maxCount <= 0 || ring.isEmpty()) return emptyList()
        val take = minOf(maxCount, ring.size)
        val rawItems = ArrayList<RingItem>(take)
        repeat(take) {
            rawItems.add(ring.removeLast())
        }
        rawItems.reverse()
        val out = ArrayList<RawFrame>(take)
        var i = 0
        try {
            while (i < rawItems.size) {
                val item = rawItems[i]
                i++ // item.materialize() owns releasing item on failure
                out.add(item.materialize(allocateRaw16, freeRaw16))
            }
            return out
        } catch (t: Throwable) {
            out.forEach { it.release() }
            while (i < rawItems.size) {
                rawItems[i++].release()
            }
            throw t
        }
    }

    @Synchronized
    fun clear() {
        while (ring.isNotEmpty()) {
            ring.removeFirst().release()
        }
        pairs.clear()
    }

    override fun close() = clear()

    private fun toRingItem(entry: RawEntry, result: CaptureResult): RingItem =
        if (entry.isRaw10) {
            RingItem.CompactRaw10(entry, result)
        } else {
            RingItem.PairedFrame(RawFrame(entry.data, entry.width, entry.height, entry.timestampNs, result, entry.onRelease))
        }

    private fun pushItemLocked(item: RingItem, nowNs: Long) {
        evictStale(nowNs)
        if (nowNs - item.timestampNs > maxAgeNs) {
            item.release()
            return
        }
        while (ring.size >= capacity) {
            ring.removeFirst().release()
        }
        ring.addLast(item)
    }

    companion object {
        const val DEFAULT_CAPACITY = 12
        const val DEFAULT_MAX_AGE_NS = 2_000_000_000L
    }
}
