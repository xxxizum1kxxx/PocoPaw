package com.izum1k.pocopaw.pipeline

import android.util.Log
import java.io.Closeable
import java.util.concurrent.Executors
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext

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
) : Closeable {
    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "burst-stream") }
    private val worker = executor.asCoroutineDispatcher()
    private val reference = CompletableDeferred<Int>()

    // worker thread only
    private val frames = ArrayList<RawFrame>()
    private var native: NativeBurst? = null
    private var failure: Throwable? = null
    private var ended = false
    private var handedOver = false
    private var refIndex = -1

    /** Takes [frame] (frames in capture order, from one thread); [metadata] is the burst's, the same for every frame. */
    fun add(frame: RawFrame, metadata: RawMetadata) {
        executor.execute {
            frames += frame
            if (failure != null) {
                frame.release()
                return@execute
            }
            try {
                val n = native ?: NativeBurst.forStream(metadata, frame).also { n ->
                    native = n
                    n.useGpu(gpu)
                    noise?.let(n::setNoiseCurve)
                    if (metadata.cropZoom > 1f) n.crop(metadata.cropZoom)
                    n.beginStream(options)
                }
                settle(n.addFrame(frame))
            } catch (e: Exception) {
                fail(e)
            }
        }
    }

    /** Index into the frames given to [add] of the frame merged onto; waits for the last frame when the burst was short. */
    suspend fun reference(): Int =
        if (reference.isCompleted) reference.await() else withContext(worker) { end(); reference.await() }

    /** After the last [add]: the merged session, which the caller closes; [NativeBurst.merge] only picks the demosaic. */
    suspend fun finish(): NativeBurst = withContext(worker) {
        end()
        handedOver = true
        checkNotNull(native)
    }

    override fun close() {
        executor.execute {
            if (!handedOver) native?.close()
            native = null
            frames.forEach { it.release() }
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
        } catch (e: Exception) {
            fail(e)
            throw e
        }
    }

    private fun settle(ref: Int) {
        if (ref < 0) return
        refIndex = ref
        reference.complete(ref)
        frames.forEachIndexed { i, f -> if (i != ref) f.release() }
    }

    private fun fail(e: Exception) {
        Log.w(TAG, "streamed merge failed", e)
        failure = e
        reference.completeExceptionally(e)
        native?.close()
        native = null
        // stages after ReferenceStage may still read the reference
        frames.forEachIndexed { i, f -> if (i != refIndex) f.release() }
    }

    private companion object {
        const val TAG = "PocoPawPipeline"
    }
}
