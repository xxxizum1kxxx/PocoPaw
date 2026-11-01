package com.izum1k.pocopaw.feature.p1

import android.graphics.SurfaceTexture
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.izum1k.pocopaw.AppContainer
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.CaptureSpec
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LensFacing
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.PreviewControls
import com.izum1k.pocopaw.data.SceneMode
import com.izum1k.pocopaw.feature.camera.CameraViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class P1Result(val folder: String, val iso: Int?, val exposureNs: Long?, val gyroRms: Double)

enum class P1Message { METERING, FAILED }

data class P1CaptureUiState(
    val camera: CameraInfo? = null,
    val variant: Int = 0,
    /** Finished bursts per variant. */
    val counts: List<Int> = emptyList(),
    val busy: Boolean = false,
    val written: Int = 0,
    val total: Int = 0,
    val last: P1Result? = null,
    val message: P1Message? = null,
    val shutterTick: Int = 0,
)

/**
 * Bare capture for one CAPTURE_P1 item: auto preview for metering, then the variant's manual RAW
 * burst written to disk in probe's format. No pipeline runs.
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
            refreshCounts()
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

    fun selectVariant(index: Int) {
        if (_ui.value.busy) return
        _ui.update { it.copy(variant = index.coerceIn(item.variants.indices)) }
    }

    fun tapFocus(x: Float, y: Float) {
        val cam = _ui.value.camera ?: return
        if (_ui.value.busy) return
        val (u, v) = CameraViewModel.viewToSensor(x, y, cam.sensorOrientation, cam.facing == LensFacing.FRONT)
        controller.focusAt(u, v)
    }

    fun consumeMessage() = _ui.update { it.copy(message = null) }

    fun shutter() {
        val s = _ui.value
        val cam = s.camera ?: return
        if (s.busy || cameraState.value !is CameraController.State.Previewing) return
        val variant = item.variants[s.variant]
        val meter = live.value
        val rms = motion.rms()?.toDouble()
        val plan = P1Exposure.plan(variant, meter, cam, rms ?: DEFAULT_GYRO_RMS)
        if (plan == null) {
            _ui.update { it.copy(message = P1Message.METERING) }
            return
        }
        _ui.update { it.copy(busy = true, written = 0, total = plan.size, message = null, shutterTick = it.shutterTick + 1) }
        viewModelScope.launch {
            val dir = withContext(Dispatchers.IO) { store.newSet(item, variant) }
            val writer = ProbeWriter(dir)
            try {
                for (chunk in plan.chunked(CHUNK_FRAMES)) {
                    val burst = controller.capture(CaptureSpec(frameCount = chunk.size, exposure = ExposurePlan.PerFrame(chunk)))
                    try {
                        for (frame in burst.frames.sortedBy { it.timestampNs }) {
                            withContext(Dispatchers.IO) { writer.write(frame) }
                            _ui.update { it.copy(written = writer.written) }
                        }
                    } finally {
                        burst.close()
                    }
                }
                withContext(Dispatchers.IO) { writer.finish(item, variant, cam, meter, rms ?: 0.0) }
                _ui.update { it.copy(last = P1Result(dir.name, writer.firstIso, writer.firstExposureNs, rms ?: 0.0)) }
            } catch (e: Exception) {
                Log.w(TAG, "P1 capture ${dir.name} failed", e)
                withContext(Dispatchers.IO) { dir.deleteRecursively() }
                _ui.update { it.copy(message = P1Message.FAILED) }
            } finally {
                controller.resetFocus()
                _ui.update { it.copy(busy = false) }
                refreshCounts()
            }
        }
    }

    private suspend fun refreshCounts() {
        val counts = withContext(Dispatchers.IO) { item.variants.map { store.count(item, it) } }
        _ui.update { it.copy(counts = counts) }
    }

    /** The viewfinder's night scene shares the meter. */
    private fun releaseMotion() {
        if (container.settings.settings.value.scene != SceneMode.NIGHT) motion.stop()
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
