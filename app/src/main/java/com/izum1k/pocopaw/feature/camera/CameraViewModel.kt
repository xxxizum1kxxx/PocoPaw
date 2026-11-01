package com.izum1k.pocopaw.feature.camera

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaActionSound
import android.net.Uri
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
import com.izum1k.pocopaw.camera.ManualExposure
import com.izum1k.pocopaw.camera.PreviewControls
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.data.CameraSettings
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.pipeline.PipelineRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.abs

enum class CaptureMode { PHOTO, PRO }

enum class Busy { IDLE, COUNTDOWN, CAPTURING, PROCESSING }

enum class CaptureMessage { CAPTURE_FAILED, SAVE_FAILED }

/** A focus tap in normalised viewfinder (texture) coordinates; [id] restarts the ring's animation. */
data class FocusPoint(val x: Float, val y: Float, val id: Long)

data class LastCapture(val uri: Uri, val thumbnail: Bitmap?)

data class CameraUiState(
    val cameras: List<CameraInfo> = emptyList(),
    val current: CameraInfo? = null,
    val mode: CaptureMode = CaptureMode.PHOTO,
    val zoom: Float = 1f,
    val evIndex: Int = 0,
    val manualIso: Int = 100,
    val manualExposureNs: Long = 10_000_000L,
    val focus: FocusPoint? = null,
    val countdown: Int? = null,
    val busy: Busy = Busy.IDLE,
    val progress: Float = 0f,
    val shutterTick: Int = 0,
    val last: LastCapture? = null,
    val message: CaptureMessage? = null,
    val messageId: Long = 0,
) {
    val backCameras: List<CameraInfo> get() = cameras.filter { it.facing == LensFacing.BACK }.sortedBy { it.relativeZoom }
    val canFlip: Boolean get() = cameras.any { it.facing == LensFacing.FRONT } && cameras.any { it.facing == LensFacing.BACK }
}

class CameraViewModel(private val container: AppContainer) : ViewModel() {
    private val controller = container.controller
    private val settingsRepo = container.settings

    val settings: StateFlow<CameraSettings> = settingsRepo.settings
    val cameraState: StateFlow<CameraController.State> = controller.state
    val live: StateFlow<LiveExposure> = controller.live
    val rawRemaining: StateFlow<Int> = controller.budget.remaining
    val rawLimit: Int = controller.budget.limit

    private val _ui = MutableStateFlow(CameraUiState())
    val ui: StateFlow<CameraUiState> = _ui.asStateFlow()

    private var texture: SurfaceTexture? = null
    private var resumed = false
    private var permitted = false
    private var focusResetJob: Job? = null
    private var shutterJob: Job? = null
    private var focusSeq = 0L
    private val sound = MediaActionSound().apply { load(MediaActionSound.SHUTTER_CLICK) }

    init {
        val catalog = container.catalog
        viewModelScope.launch {
            val cameras = withContext(Dispatchers.IO) { catalog.cameras }
            val initial = catalog.byId(settings.value.lastCameraId) ?: catalog.default()
            _ui.update { it.copy(cameras = cameras, current = initial) }
            maybeOpen()
            container.output.latest()?.let { uri ->
                if (_ui.value.last == null) _ui.update { it.copy(last = LastCapture(uri, container.output.thumbnail(uri))) }
            }
        }
    }

    fun onPermission(granted: Boolean) {
        permitted = granted
        maybeOpen()
    }

    fun onResume() {
        resumed = true
        maybeOpen()
    }

    fun onPause() {
        resumed = false
        shutterJob?.takeIf { _ui.value.busy == Busy.COUNTDOWN }?.cancel()
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
        val cam = _ui.value.current ?: return
        if (!resumed || !permitted) return
        viewModelScope.launch { controller.open(cam, t, previewControls()) }
    }

    fun selectCamera(info: CameraInfo) {
        if (info.id == _ui.value.current?.id || _ui.value.busy != Busy.IDLE) return
        focusResetJob?.cancel()
        _ui.update { it.copy(current = info, zoom = 1f, evIndex = 0, focus = null) }
        viewModelScope.launch { settingsRepo.setLastCameraId(info.id) }
        maybeOpen()
    }

    fun flip() {
        val s = _ui.value
        val target = if (s.current?.facing == LensFacing.FRONT) {
            s.backCameras.firstOrNull { it.relativeZoom == 1f } ?: s.backCameras.firstOrNull()
        } else {
            s.cameras.firstOrNull { it.facing == LensFacing.FRONT }
        }
        target?.let(::selectCamera)
    }

    fun setMode(mode: CaptureMode) {
        val s = _ui.value
        if (mode == s.mode) return
        val next = if (mode == CaptureMode.PRO) {
            val l = live.value
            val cam = s.current
            s.copy(
                mode = mode,
                manualIso = snapIso(l.iso ?: s.manualIso, cam),
                manualExposureNs = snapShutter(l.exposureNs ?: s.manualExposureNs, cam),
            )
        } else {
            s.copy(mode = mode)
        }
        _ui.value = next
        pushControls()
    }

    fun zoomBy(factor: Float) {
        val cam = _ui.value.current ?: return
        val z = (_ui.value.zoom * factor).coerceIn(cam.zoomRange.lower, cam.zoomRange.upper.coerceAtMost(MAX_ZOOM))
        if (abs(z - _ui.value.zoom) < 0.005f) return
        _ui.update { it.copy(zoom = z) }
        pushControls()
    }

    fun setEv(index: Int) {
        val cam = _ui.value.current ?: return
        val i = index.coerceIn(cam.evRange.lower, cam.evRange.upper)
        if (i == _ui.value.evIndex) return
        _ui.update { it.copy(evIndex = i) }
        pushControls()
    }

    fun setIso(iso: Int) {
        _ui.update { it.copy(manualIso = iso) }
        pushControls()
    }

    fun setShutter(ns: Long) {
        _ui.update { it.copy(manualExposureNs = ns) }
        pushControls()
    }

    /** A tap at normalised texture coordinates (portrait, as displayed). */
    fun tapFocus(x: Float, y: Float) {
        val cam = _ui.value.current ?: return
        val (u, v) = viewToSensor(x, y, cam.sensorOrientation, cam.facing == LensFacing.FRONT)
        controller.focusAt(u, v)
        _ui.update { it.copy(focus = FocusPoint(x, y, ++focusSeq)) }
        focusResetJob?.cancel()
        focusResetJob = viewModelScope.launch {
            delay(FOCUS_HOLD_MS)
            _ui.update { it.copy(focus = null) }
            controller.resetFocus()
        }
    }

    fun setFlash(v: FlashMode) = viewModelScope.launch {
        settingsRepo.setFlash(v)
        pushControls(v)
    }

    fun setTimer(v: TimerMode) = viewModelScope.launch { settingsRepo.setTimer(v) }
    fun setAspect(v: AspectRatio) = viewModelScope.launch { settingsRepo.setAspect(v) }
    fun setGrid(v: Boolean) = viewModelScope.launch { settingsRepo.setGrid(v) }

    fun consumeMessage() {
        _ui.update { it.copy(message = null) }
    }

    /** Shutter press: optional countdown, then capture and pipeline. A press during the countdown cancels it. */
    fun shutter(deviceRotation: Int) {
        val s = _ui.value
        if (s.busy == Busy.COUNTDOWN) {
            shutterJob?.cancel()
            _ui.update { it.copy(busy = Busy.IDLE, countdown = null) }
            return
        }
        if (s.busy != Busy.IDLE || cameraState.value !is CameraController.State.Previewing) return
        shutterJob = viewModelScope.launch {
            val timer = settings.value.timer.seconds
            if (timer > 0) {
                _ui.update { it.copy(busy = Busy.COUNTDOWN) }
                for (n in timer downTo 1) {
                    _ui.update { it.copy(countdown = n) }
                    delay(1_000)
                }
                _ui.update { it.copy(countdown = null) }
            }
            capture(deviceRotation)
        }
    }

    private suspend fun capture(deviceRotation: Int) {
        val s = _ui.value
        val pipeline = PipelineRegistry.create(PipelineRegistry.defaultId)
        _ui.update { it.copy(busy = Busy.CAPTURING, progress = 0f, shutterTick = it.shutterTick + 1) }
        if (settings.value.shutterSound) sound.play(MediaActionSound.SHUTTER_CLICK)
        val spec = CaptureSpec(
            frameCount = pipeline.framesPerCapture,
            exposure = exposurePlan(s),
            flash = settings.value.flash,
            deviceRotation = deviceRotation,
            zoomRatio = s.zoom,
        )
        val burst = try {
            controller.capture(spec)
        } catch (e: Exception) {
            Log.w(TAG, "capture failed", e)
            fail(CaptureMessage.CAPTURE_FAILED)
            return
        }
        _ui.update { it.copy(busy = Busy.PROCESSING) }
        try {
            val result = pipeline.process(burst, container.output) { p -> _ui.update { it.copy(progress = p) } }
            val uri = result.primaryUri
            _ui.update {
                it.copy(
                    busy = Busy.IDLE,
                    last = if (uri != null) LastCapture(uri, result.thumbnail) else it.last,
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "pipeline failed", e)
            fail(CaptureMessage.SAVE_FAILED)
        }
    }

    private fun fail(message: CaptureMessage) {
        _ui.update { it.copy(busy = Busy.IDLE, countdown = null, message = message, messageId = it.messageId + 1) }
    }

    private fun exposurePlan(s: CameraUiState): ExposurePlan = when (s.mode) {
        CaptureMode.PHOTO -> ExposurePlan.Auto(s.evIndex)
        CaptureMode.PRO -> ExposurePlan.Manual(ManualExposure(s.manualIso, s.manualExposureNs))
    }

    private fun previewControls(flash: FlashMode = settings.value.flash): PreviewControls {
        val s = _ui.value
        return PreviewControls(exposure = exposurePlan(s), flash = flash, zoomRatio = s.zoom)
    }

    private fun pushControls(flash: FlashMode = settings.value.flash) {
        controller.updateControls(previewControls(flash))
    }

    override fun onCleared() {
        sound.release()
    }

    companion object {
        private const val TAG = "PocoPawCamera"
        private const val FOCUS_HOLD_MS = 5_000L
        private const val MAX_ZOOM = 10f

        val ISO_STOPS = listOf(50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400)

        val SHUTTER_STOPS: List<Long> = listOf(
            8000, 4000, 2000, 1000, 500, 250, 125, 100, 60, 30, 15, 8, 4, 2,
        ).map { 1_000_000_000L / it } + listOf(1_000_000_000L)

        fun isoStops(cam: CameraInfo?): List<Int> {
            val r = cam?.isoRange ?: return ISO_STOPS
            return ISO_STOPS.filter { it in r.lower..r.upper }.ifEmpty { listOf(r.lower) }
        }

        fun shutterStops(cam: CameraInfo?): List<Long> {
            val r = cam?.exposureRangeNs ?: return SHUTTER_STOPS
            return SHUTTER_STOPS.filter { it in r.lower..r.upper }.ifEmpty { listOf(r.lower) }
        }

        fun snapIso(iso: Int, cam: CameraInfo?): Int = isoStops(cam).minBy { abs(it - iso) }

        fun snapShutter(ns: Long, cam: CameraInfo?): Long = shutterStops(cam).minBy { abs(it - ns) }

        /**
         * Displayed (portrait, mirrored for the front camera) to sensor coordinates. The preview
         * shows the sensor rotated clockwise by its orientation.
         */
        fun viewToSensor(x: Float, y: Float, sensorOrientation: Int, mirrored: Boolean): Pair<Float, Float> {
            val vx = if (mirrored) 1f - x else x
            return when (sensorOrientation) {
                90 -> y to 1f - vx
                180 -> 1f - vx to 1f - y
                270 -> 1f - y to vx
                else -> vx to y
            }
        }
    }
}
