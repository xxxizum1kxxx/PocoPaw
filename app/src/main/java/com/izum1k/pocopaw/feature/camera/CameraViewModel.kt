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
import com.izum1k.pocopaw.data.OutputFormat
import com.izum1k.pocopaw.data.Processing
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.data.SceneMode
import com.izum1k.pocopaw.pipeline.BurstLook
import com.izum1k.pocopaw.pipeline.BurstScene
import com.izum1k.pocopaw.pipeline.DngPassthroughPipeline
import com.izum1k.pocopaw.pipeline.LensCalibration
import com.izum1k.pocopaw.pipeline.HdrPlusPipeline
import com.izum1k.pocopaw.pipeline.ImagePipeline
import com.izum1k.pocopaw.pipeline.PipelineOptions
import com.izum1k.pocopaw.pipeline.PipelineRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.roundToLong

enum class CaptureMode { PHOTO, PRO }

enum class Busy { IDLE, COUNTDOWN, CAPTURING, PROCESSING }

enum class CaptureMessage { CAPTURE_FAILED, SAVE_FAILED }

/** A focus tap in normalised viewfinder (texture) coordinates; [id] restarts the ring's animation. */
data class FocusPoint(val x: Float, val y: Float, val id: Long)

data class LastCapture(val uri: Uri, val thumbnail: Bitmap?)

/** The last pipeline run, for the developer chip. */
data class PipelineRun(val pipelineId: String, val frames: Int, val millis: Long)

data class CameraUiState(
    val cameras: List<CameraInfo> = emptyList(),
    val current: CameraInfo? = null,
    val mode: CaptureMode = CaptureMode.PHOTO,
    val zoom: Float = 1f,
    val evIndex: Int = 0,
    val manualIso: Int = 100,
    val manualExposureNs: Long = 10_000_000L,
    /** Pro: AE picks the ISO (with a manual shutter, the one that keeps AE's exposure). */
    val isoAuto: Boolean = true,
    /** Pro: AE picks the shutter (with a manual ISO, the one that keeps AE's exposure). */
    val shutterAuto: Boolean = true,
    val focus: FocusPoint? = null,
    val countdown: Int? = null,
    val busy: Busy = Busy.IDLE,
    val progress: Float = 0f,
    val shutterTick: Int = 0,
    val last: LastCapture? = null,
    val lastRun: PipelineRun? = null,
    val message: CaptureMessage? = null,
    val messageId: Long = 0,
) {
    val backCameras: List<CameraInfo> get() = cameras.filter { it.facing == LensFacing.BACK }.sortedBy { it.relativeZoom }
    val canFlip: Boolean get() = cameras.any { it.facing == LensFacing.FRONT } && cameras.any { it.facing == LensFacing.BACK }
}

class CameraViewModel(private val container: AppContainer) : ViewModel() {
    private val controller = container.controller
    private val motion = container.motion
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
            settings.collect { if (it.scene == SceneMode.NIGHT) motion.start() else motion.stop() }
        }
        // A setting that asks for more frames than are left (8 to 16, DNG to a burst) reopens
        // right away instead of at the next press.
        viewModelScope.launch {
            combine(settings, cameraState) { _, state -> state }.collect { state ->
                if (state is CameraController.State.Previewing && _ui.value.busy == Busy.IDLE && !controller.budget.canAfford(nextFrames())) {
                    controller.reopen()
                }
            }
        }
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

    /** A manual ISO, or null for auto. */
    fun setIso(iso: Int?) {
        _ui.update { if (iso == null) it.copy(isoAuto = true) else it.copy(isoAuto = false, manualIso = iso) }
        pushControls()
    }

    /** A manual shutter, or null for auto. */
    fun setShutter(ns: Long?) {
        _ui.update { if (ns == null) it.copy(shutterAuto = true) else it.copy(shutterAuto = false, manualExposureNs = ns) }
        pushControls()
    }

    /** Reopens the camera to give back its RAW budget. */
    fun restartCamera() {
        if (_ui.value.busy != Busy.IDLE) return
        viewModelScope.launch { controller.reopen() }
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
        val cam = s.current ?: return
        val pipeline = pipelineFor(settings.value, cam)
        _ui.update { it.copy(busy = Busy.CAPTURING, progress = 0f, shutterTick = it.shutterTick + 1) }
        if (settings.value.shutterSound) sound.play(MediaActionSound.SHUTTER_CLICK)
        val spec = CaptureSpec(
            frameCount = pipeline.framesPerCapture,
            exposure = pipeline.exposureFor(captureExposure(s, cam), live.value.copy(motionRadPerSec = motion.rms()), cam),
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
        // The burst is in memory now; reopening while it merges keeps the next press from waiting.
        if (!controller.budget.canAfford(nextFrames())) viewModelScope.launch { controller.reopen() }
        try {
            val result = pipeline.process(burst, container.output) { p -> _ui.update { it.copy(progress = p) } }
            val uri = result.primaryUri
            _ui.update {
                it.copy(
                    busy = Busy.IDLE,
                    last = if (uri != null) LastCapture(uri, result.thumbnail) else it.last,
                    lastRun = PipelineRun(pipeline.id, burst.size, result.totalMillis),
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "pipeline failed", e)
            fail(CaptureMessage.SAVE_FAILED)
        } finally {
            burst.close()
        }
    }

    /**
     * DNG output keeps the single-frame pass-through; JPEG output runs the burst pipeline. A flash
     * fires for one frame only, so a flash capture is a burst of one.
     */
    private fun pipelineFor(cfg: CameraSettings, cam: CameraInfo): ImagePipeline {
        if (cfg.output == OutputFormat.DNG) return PipelineRegistry.create(DngPassthroughPipeline.ID)
        val id = when (cfg.processing) {
            Processing.HDR_PLUS -> HdrPlusPipeline.ID
            Processing.HDR_PLUS_KR -> HdrPlusPipeline.ID_KR
        }
        val flash = cam.hasFlash && cfg.flash != FlashMode.OFF
        return PipelineRegistry.create(
            id,
            PipelineOptions(
                frames = if (flash) 1 else cfg.burstFrames.count,
                writeJpeg = true,
                writeDng = cfg.output == OutputFormat.DNG_JPEG,
                look = BurstLook(style = cfg.look.preset),
                scene = if (flash) BurstScene.AUTO else when (cfg.scene) {
                    SceneMode.AUTO -> BurstScene.AUTO
                    SceneMode.BRACKET -> BurstScene.BRACKET
                    SceneMode.NIGHT -> BurstScene.NIGHT
                },
                ultraHdr = cfg.ultraHdr,
                lca = LensCalibration.lca(cam.id),
            ),
        )
    }

    /** RAW frames the next press will ask for. */
    private fun nextFrames(): Int = _ui.value.current?.let { pipelineFor(settings.value, it).framesPerCapture } ?: 1

    private fun fail(message: CaptureMessage) {
        _ui.update { it.copy(busy = Busy.IDLE, countdown = null, message = message, messageId = it.messageId + 1) }
    }

    /** What the preview runs: Camera2 has no half-auto, so unless both are manual AE keeps metering. */
    private fun exposurePlan(s: CameraUiState): ExposurePlan = when {
        s.mode == CaptureMode.PHOTO -> ExposurePlan.Auto(s.evIndex)
        s.isoAuto || s.shutterAuto -> ExposurePlan.Auto()
        else -> ExposurePlan.Manual(ManualExposure(s.manualIso, s.manualExposureNs))
    }

    private fun captureExposure(s: CameraUiState, cam: CameraInfo): ExposurePlan {
        if (s.mode != CaptureMode.PRO || s.isoAuto == s.shutterAuto) return exposurePlan(s)
        return priorityExposure(s, live.value, cam)?.let { ExposurePlan.Manual(it) } ?: ExposurePlan.Auto()
    }

    private fun previewControls(flash: FlashMode = settings.value.flash): PreviewControls {
        val s = _ui.value
        return PreviewControls(exposure = exposurePlan(s), flash = flash, zoomRatio = s.zoom)
    }

    private fun pushControls(flash: FlashMode = settings.value.flash) {
        controller.updateControls(previewControls(flash))
    }

    override fun onCleared() {
        motion.stop()
        sound.release()
    }

    companion object {
        private const val TAG = "PocoPawCamera"
        private const val FOCUS_HOLD_MS = 5_000L
        private const val MAX_ZOOM = 10f

        val ISO_STOPS = listOf(50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400)

        /** Third stops, as a camera's dial marks them: evenly spaced in exposure, not in time. */
        val SHUTTER_STOPS: List<Long> = listOf(
            8000, 6400, 5000, 4000, 3200, 2500, 2000, 1600, 1250, 1000, 800, 640, 500, 400, 320, 250, 200, 160,
            125, 100, 80, 60, 50, 40, 30, 25, 20, 15, 13, 10, 8, 6, 5, 4,
        ).map { 1_000_000_000L / it } + listOf(0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 15.0, 20.0, 25.0, 30.0)
            .map { (it * 1e9).roundToLong() }

        fun isoStops(cam: CameraInfo?): List<Int> {
            val r = cam?.isoRange ?: return ISO_STOPS
            return ISO_STOPS.filter { it in r.lower..r.upper }.ifEmpty { listOf(r.lower) }
        }

        fun shutterStops(cam: CameraInfo?): List<Long> {
            val r = cam?.exposureRangeNs ?: return SHUTTER_STOPS
            return SHUTTER_STOPS.filter { it in r.lower..r.upper }.ifEmpty { listOf(r.lower) }
        }

        fun snapIso(iso: Int, cam: CameraInfo?): Int = isoStops(cam).minBy { abs(it - iso) }

        fun snapShutter(ns: Long, cam: CameraInfo?): Long = shutterStops(cam).minBy { abs(ln(it.toDouble() / ns)) }

        /**
         * Pro with one of the two on auto: the other one is the user's, and the auto one is
         * whatever keeps the preview AE's ISO × exposure. Null until AE has reported.
         */
        fun priorityExposure(s: CameraUiState, live: LiveExposure, cam: CameraInfo?): ManualExposure? {
            val iso0 = live.iso ?: return null
            val exp0 = live.exposureNs ?: return null
            val product = iso0.toDouble() * exp0
            return when {
                s.isoAuto && !s.shutterAuto -> {
                    val r = cam?.isoRange
                    val iso = (product / s.manualExposureNs).roundToInt()
                    ManualExposure(if (r != null) iso.coerceIn(r.lower, r.upper) else iso, s.manualExposureNs)
                }
                s.shutterAuto && !s.isoAuto -> {
                    val r = cam?.exposureRangeNs
                    val exp = (product / s.manualIso).roundToLong()
                    ManualExposure(s.manualIso, if (r != null) exp.coerceIn(r.lower, r.upper) else exp)
                }
                else -> ManualExposure(iso0, exp0)
            }
        }

        /** The auto half of [priorityExposure] hit its range and misses AE's exposure by more than a sixth of a stop. */
        fun priorityClipped(s: CameraUiState, live: LiveExposure, cam: CameraInfo?): Boolean {
            val m = priorityExposure(s, live, cam) ?: return false
            val product = (live.iso ?: return false).toDouble() * (live.exposureNs ?: return false)
            return abs(ln(m.iso.toDouble() * m.exposureNs / product)) > ln(2.0) / 6
        }

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
