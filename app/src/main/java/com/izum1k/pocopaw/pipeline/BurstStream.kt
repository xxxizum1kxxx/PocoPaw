package com.izum1k.pocopaw.pipeline

import android.util.Log
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext

/** Per-frame streaming quality telemetry from `BurstSession::add_frame_progress`. */
data class StreamProgress(
    val refIdx: Int,
    val acceptedCount: Int,
    val neff: Float,
    val phaseCoverage: Float,
    val estimatedSnrGainDb: Float,
    val shouldStop: Boolean,
)

/** Abstraction of a native streaming session so [BurstStream] is JVM-testable without `libburstpipe.so`. */
interface StreamSession : Closeable {
    fun useGpu(enabled: Boolean)
    fun setAccelBackend(backend: AccelBackend)
    fun setNoiseCurve(curve: NoiseCurve)
    fun crop(zoom: Float)
    fun beginStream(options: MergeOptions)
    fun addFrameWithProgress(frame: RawFrame): StreamProgress
    fun endFrames(): Int
}

/**
 * Aligns and merges a burst while it is being captured (the classic spatial merge of one exposure,
 * `BurstSession::add_frame`). The camera hands each frame to [add] as it arrives; one background thread
 * feeds them to a [NativeBurst.forStream] session, and every frame but the reference is released once
 * merged, so memory does not grow with the burst. The pipeline then takes the reference ([reference])
 * and the merged session ([finish]); closing the [RawBurst] closes the stream.
 */
class BurstStream(
    private val gpu: Boolean,
    private val noise: NoiseCurve?,
    private val options: MergeOptions,
    private val sessionFactory: (RawMetadata, RawFrame) -> StreamSession = { m, f -> NativeBurst.forStream(m, f) },
) : Closeable {
    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "burst-stream") }
    private val worker = executor.asCoroutineDispatcher()
    private val reference = CompletableDeferred<Int>()
    private val closed = AtomicBoolean(false)

    // worker thread only
    private val frames = ArrayList<RawFrame>()
    private var native: StreamSession? = null
    private var failure: Throwable? = null
    private var ended = false
    private var handedOver = false
    private var refIndex = -1

    @Volatile
    var shouldStop: Boolean = false
        private set

    @Volatile
    var latestProgress: StreamProgress? = null
        private set

    val earlyStopEnabled: Boolean get() = options.earlyStopEnabled

    /** Takes [frame] (frames in capture order, from one thread); [metadata] is the burst's, the same for every frame. */
    fun add(frame: RawFrame, metadata: RawMetadata) {
        if (closed.get()) {
            frame.release()
            return
        }
        try {
            executor.execute {
                if (closed.get() || failure != null) {
                    frame.release()
                    return@execute
                }
                frames += frame
                try {
                    val n = native ?: sessionFactory(metadata, frame).also { n ->
                        native = n
                        n.useGpu(gpu)
                        if (options.accelBackend != AccelBackend.AUTO) n.setAccelBackend(options.accelBackend)
                        noise?.let(n::setNoiseCurve)
                        if (metadata.cropZoom > 1f) n.crop(metadata.cropZoom)
                        n.beginStream(options)
                    }
                    val progress = n.addFrameWithProgress(frame)
                    latestProgress = progress
                    if (progress.shouldStop) shouldStop = true
                    settle(progress.refIdx)
                } catch (t: Throwable) {
                    fail(t)
                }
            }
        } catch (_: RejectedExecutionException) {
            frame.release()
        }
    }

    /** Waits for all queued [add] calls on the stream thread and returns the latest [StreamProgress]. */
    suspend fun awaitProgress(): StreamProgress? = withContext(worker) {
        failure?.let { throw it }
        latestProgress
    }

    /** Index into the frames given to [add] of the frame merged onto; waits for the last frame when the burst was short. */
    suspend fun reference(): Int =
        if (reference.isCompleted) reference.await() else withContext(worker) { end(); reference.await() }

    /** After the last [add]: the merged session, which the caller closes; [NativeBurst.merge] only picks the demosaic. */
    suspend fun finish(): NativeBurst = finishSession() as NativeBurst

    /** JVM-testable variant returning the [StreamSession] after finalizing [end]. */
    suspend fun finishSession(): StreamSession = withContext(worker) {
        end()
        handedOver = true
        checkNotNull(native)
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        try {
            executor.execute {
                if (!handedOver) native?.close()
                native = null
                frames.forEach { it.release() }
                frames.clear()
            }
        } catch (_: RejectedExecutionException) {
            // executor already shut down
        }
        executor.shutdown()
    }

    private fun end() {
        failure?.let { throw it }
        if (ended) return
        ended = true
        val n = native ?: throw IllegalStateException("no frames streamed")
        try {
            settle(n.endFrames())
        } catch (t: Throwable) {
            fail(t)
            throw t
        }
    }

    private fun settle(ref: Int) {
        if (ref < 0) return
        refIndex = ref
        reference.complete(ref)
        frames.forEachIndexed { i, f -> if (i != ref) f.release() }
    }

    private fun fail(t: Throwable) {
        Log.w(TAG, "streamed merge failed", t)
        failure = t
        reference.completeExceptionally(t)
        native?.close()
        native = null
        // stages after ReferenceStage may still read the reference
        frames.forEachIndexed { i, f -> if (i != refIndex) f.release() }
    }

    private companion object {
        const val TAG = "PocoPawPipeline"
    }
}
