package com.izum1k.pocopaw.pipeline

import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class ProcessedCapture(val uri: Uri, val mimeType: String, val thumbnail: Bitmap?)

data class PipelineRunSummary(val pipelineId: String, val frames: Int, val millis: Long)

data class ProcessingQueueState(
    val pending: Int = 0,
    val progress: Float = 0f,
    val last: ProcessedCapture? = null,
    val lastRun: PipelineRunSummary? = null,
    val errorSeq: Long = 0,
) {
    val active: Boolean get() = pending > 0
    val isFull: Boolean get() = pending >= ProcessingQueue.MAX_PENDING
}

/**
 * Serializes pipeline runs off the ViewModel so leaving the camera keeps processing,
 * and allows up to [MAX_PENDING] bursts in flight before gating the shutter.
 */
class ProcessingQueue(
    private val scope: CoroutineScope,
    private val sink: OutputSink,
) {
    private val mutex = Mutex()
    private val _state = MutableStateFlow(ProcessingQueueState())
    val state: StateFlow<ProcessingQueueState> = _state.asStateFlow()

    val isFull: Boolean get() = _state.value.isFull

    fun enqueue(pipeline: ImagePipeline, burst: RawBurst) {
        _state.update { it.copy(pending = it.pending + 1) }
        scope.launch {
            try {
                mutex.withLock {
                    _state.update { it.copy(progress = 0f) }
                    val result = pipeline.process(burst, sink) { p ->
                        _state.update { it.copy(progress = p) }
                    }
                    val primary = result.primaryOutput
                    _state.update { s ->
                        s.copy(
                            last = if (primary != null) ProcessedCapture(primary.uri, primary.kind.mimeType, result.thumbnail) else s.last,
                            lastRun = PipelineRunSummary(pipeline.id, burst.size, result.totalMillis),
                        )
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "pipeline failed", e)
                _state.update { it.copy(errorSeq = it.errorSeq + 1) }
            } finally {
                burst.close()
                _state.update { s ->
                    val next = (s.pending - 1).coerceAtLeast(0)
                    s.copy(pending = next, progress = if (next == 0) 0f else s.progress)
                }
            }
        }
    }

    companion object {
        const val MAX_PENDING = 2
        private const val TAG = "PocoPawProcessing"
    }
}
