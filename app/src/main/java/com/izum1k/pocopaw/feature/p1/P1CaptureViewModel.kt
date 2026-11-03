package com.izum1k.pocopaw.feature.p1

import android.graphics.SurfaceTexture
import android.hardware.camera2.CaptureResult
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.izum1k.pocopaw.AppContainer
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.CaptureSpec
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.PreviewControls
import com.izum1k.pocopaw.feature.camera.CameraViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class P1Result(val dir: File, val frames: Int, val iso: Int?, val exposureNs: Long?, val gyroRms: Double)

enum class P1Message { METERING, FAILED, DISCARDED }

data class P1CaptureUiState(
    val camera: CameraInfo? = null,
    /** Kept bursts of this item. */
    val count: Int = 0,
    val busy: Boolean = false,
    val written: Int = 0,
    val total: Int = 0,
    /** A finished burst waiting to be kept or discarded. */
    val pending: P1Result? = null,
    val last: P1Result? = null,
    val message: P1Message? = null,
)

/**
 * Fully automatic capture for one P1 capture list item: auto preview (AE, AF, AWB) for metering, then
 * the item's manual RAW burst written to disk in probe's format and offered for review. No
 * pipeline runs.
 */
class P1CaptureViewModel(private val container: AppContainer, private val store: P1Store, val item: P1Item) : ViewModel() {
    private val controller = container.controller
    private val motion = container.motion

    val cameraState: StateFlow<CameraController.State> = controller.state
    val live: StateFlow<LiveExposure> = controller.live

    private val _ui = MutableStateFlow(P1CaptureUiState())
    val ui: StateFlow<P1CaptureUiState> = _ui.asStateFlow()

    private var texture: SurfaceTexture? = null
    private var resumed = false

    init {
        motion.start()
        viewModelScope.launch {
            val cam = withContext(Dispatchers.IO) { container.catalog.byId(item.cameraId) ?: container.catalog.default() }
            _ui.update { it.copy(camera = cam) }
            refreshCount()
            maybeOpen()
        }
    }

    fun onResume() {
        resumed = true
        motion.start()
        maybeOpen()
    }

    fun onPause() {
        resumed = false
        releaseMotion()
        viewModelScope.launch { controller.close() }
    }

    fun onSurfaceAvailable(t: SurfaceTexture) {
        texture = t
        maybeOpen()
    }

    fun onSurfaceDestroyed() {
        texture = null
        viewModelScope.launch { controller.close() }
    }

    fun retry() {
        viewModelScope.launch { controller.retry() }
    }

    private fun maybeOpen() {
        val t = texture ?: return
        val cam = _ui.value.camera ?: return
        if (!resumed) return
        viewModelScope.launch { controller.open(cam, t, PreviewControls()) }
    }

    fun consumeMessage() = _ui.update { it.copy(message = null) }

    fun keep() {
        val r = _ui.value.pending ?: return
        _ui.update { it.copy(pending = null, last = r) }
        viewModelScope.launch { refreshCount() }
    }

    fun discard() {
        val r = _ui.value.pending ?: return
        _ui.update { it.copy(pending = null, message = P1Message.DISCARDED) }
        viewModelScope.launch {
            withContext(Dispatchers.IO) { r.dir.deleteRecursively() }
            refreshCount()
        }
    }

    fun shutter() {
        val s = _ui.value
        val cam = s.camera ?: return
        if (s.busy || s.pending != null || cameraState.value !is CameraController.State.Previewing) return
        val meter = live.value
        val rms = motion.rms()?.toDouble()
        val plan = P1Exposure.plan(item.variant, meter, cam, rms ?: DEFAULT_GYRO_RMS)
        if (plan == null) {
            _ui.update { it.copy(message = P1Message.METERING) }
            return
        }
        _ui.update { it.copy(busy = true, written = 0, total = plan.size, message = null) }
        viewModelScope.launch {
            val dir = withContext(Dispatchers.IO) { store.newSet(item) }
            val writer = ProbeWriter(dir)
            try {
                var focus: Float? = null
                for (chunk in plan.chunked(CHUNK_FRAMES)) {
                    val spec = CaptureSpec(frameCount = chunk.size, exposure = ExposurePlan.PerFrame(chunk), focusDioptres = focus)
                    val burst = controller.capture(spec)
                    if (cam.hasAutoFocus && focus == null) focus = burst.frames.firstOrNull()?.result?.get(CaptureResult.LENS_FOCUS_DISTANCE)
                    try {
                        for (frame in burst.frames.sortedBy { it.timestampNs }) {
                            withContext(Dispatchers.IO) { writer.write(frame) }
                            _ui.update { it.copy(written = writer.written) }
                        }
                    } finally {
                        burst.close()
                    }
                }
                withContext(Dispatchers.IO) { writer.finish(item, cam, meter, rms ?: 0.0) }
                val result = P1Result(dir, writer.written, writer.firstIso, writer.firstExposureNs, rms ?: 0.0)
                _ui.update { it.copy(pending = result) }
            } catch (e: Exception) {
                Log.w(TAG, "P1 capture ${dir.name} failed", e)
                withContext(Dispatchers.IO) { dir.deleteRecursively() }
                _ui.update { it.copy(message = P1Message.FAILED) }
            } finally {
                _ui.update { it.copy(busy = false) }
            }
        }
    }

    private suspend fun refreshCount() {
        val n = withContext(Dispatchers.IO) { store.sets(item).size }
        _ui.update { it.copy(count = n) }
    }

    /** The viewfinder's night scene and KR share the meter. */
    private fun releaseMotion() {
        if (!CameraViewModel.needsMotion(container.settings.settings.value)) motion.stop()
    }

    override fun onCleared() = releaseMotion()

    companion object {
        private const val TAG = "PocoPawP1"

        /** Frames held in memory at once (about 32 MB each on the main camera). */
        const val CHUNK_FRAMES = 8

        /** What probe assumes without gyroscope samples. */
        private const val DEFAULT_GYRO_RMS = 0.02
    }
}
