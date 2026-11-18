package com.izum1k.pocopaw.feature.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.media.MediaActionSound
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.izum1k.pocopaw.AppContainer
import com.izum1k.pocopaw.BuildConfig
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.CaptureSpec
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LensFacing
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.PreviewControls
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.data.CameraSettings
import com.izum1k.pocopaw.data.DebugCapturePipeline
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.pipeline.BurstLook
import com.izum1k.pocopaw.pipeline.ImagePipeline
import com.izum1k.pocopaw.pipeline.LensCalibration
import com.izum1k.pocopaw.pipeline.PipelineOptions
import com.izum1k.pocopaw.pipeline.PipelineRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs

enum class Busy { IDLE, COUNTDOWN, CAPTURING, PROCESSING }

enum class CaptureMessage { CAPTURE_FAILED, SAVE_FAILED }

/** A focus tap in normalised viewfinder (texture) coordinates; [id] restarts the ring's animation. */
data class FocusPoint(val x: Float, val y: Float, val id: Long)

data class LastCapture(val uri: Uri, val mimeType: String, val thumbnail: Bitmap?)

data class CameraUiState(
    val cameras: List<CameraInfo> = emptyList(),
    val current: CameraInfo? = null,
    val zoom: Float = 1f,
    /** The main lens's 2x step: the photo is cropped to the centre 1/[cropZoom]. */
    val cropZoom: Float = 1f,
    /** Where the viewfinder animates [zoom] to (null: no animation); [zoomAnimId] restarts or cancels it. */
    val zoomTarget: Float? = null,
    val zoomAnimId: Int = 0,
    val evIndex: Int = 0,
    /** Warm (+) / cool (−) steps off auto white balance, within ±[WB_MAX_STEPS]. */
    val wbShift: Int = 0,
    val focus: FocusPoint? = null,
    /** The tap's AF/AE lock holds until released, instead of for [FOCUS_HOLD_MS]. */
    val focusHeld: Boolean = false,
    val countdown: Int? = null,
    val busy: Busy = Busy.IDLE,
    val backgroundProcessing: Boolean = false,
    val progress: Float = 0f,
    val shutterTick: Int = 0,
    val last: LastCapture? = null,
    val recentUris: List<Uri> = emptyList(),
    val awaitingGallery: Boolean = false,
    val openGalleryPending: Boolean = false,
    val galleryOpenId: Long = 0,
    val message: CaptureMessage? = null,
    val messageId: Long = 0,
) {
    val backCameras: List<CameraInfo> get() = cameras.filter { it.facing == LensFacing.BACK }.sortedBy { it.relativeZoom }
    val canFlip: Boolean get() = cameras.any { it.facing == LensFacing.FRONT } && cameras.any { it.facing == LensFacing.BACK }
}

class CameraViewModel(
    private val appContext: Context,
    private val container: AppContainer,
) : ViewModel() {
    private val controller = container.controller
    private val motion = container.motion
    private val settingsRepo = container.settings

    val settings: StateFlow<CameraSettings> = settingsRepo.settings
    val cameraState: StateFlow<CameraController.State> = controller.state
    val live: StateFlow<LiveExposure> = controller.live

    private val _ui = MutableStateFlow(CameraUiState())
    val ui: StateFlow<CameraUiState> = _ui.asStateFlow()

    private var texture: SurfaceTexture? = null
    private var resumed = false
    private var permitted = false
    private var focusResetJob: Job? = null
    private var shutterJob: Job? = null
    private var refreshJob: Job? = null
    private var focusSeq = 0L
    private val sound = MediaActionSound().apply { load(MediaActionSound.SHUTTER_CLICK) }

    init {
        val catalog = container.catalog
        viewModelScope.launch {
            val cameras = withContext(Dispatchers.IO) { catalog.cameras }
            val initial = catalog.byId(settings.value.lastCameraId) ?: catalog.default()
            _ui.update { it.copy(cameras = cameras, current = initial) }
            maybeOpen()
            refreshRecentCaptures()
        }
        viewModelScope.launch {
            var seenErrors = container.processing.state.value.errorSeq
            container.processing.state.collect { st ->
                val hadError = st.errorSeq > seenErrors
                if (hadError) {
                    seenErrors = st.errorSeq
                    _ui.update { it.copy(message = CaptureMessage.SAVE_FAILED, messageId = it.messageId + 1) }
                }
                _ui.update { ui ->
                    val latest = st.last?.let { LastCapture(it.uri, it.mimeType, it.thumbnail) } ?: ui.last
                    val updatedRecents = if (st.last != null) {
                        (listOf(st.last.uri) + ui.recentUris).distinct()
                    } else {
                        ui.recentUris
                    }
                    val finishedWaiting = ui.awaitingGallery && !st.active
                    ui.copy(
                        backgroundProcessing = st.active,
                        progress = st.progress,
                        busy = if (ui.busy == Busy.PROCESSING && !st.isFull) Busy.IDLE else ui.busy,
                        last = latest,
                        recentUris = updatedRecents,
                        awaitingGallery = ui.awaitingGallery && st.active && !hadError,
                        openGalleryPending = if (finishedWaiting && !hadError) true else ui.openGalleryPending,
                        galleryOpenId = if (finishedWaiting && !hadError) ui.galleryOpenId + 1 else ui.galleryOpenId,
                    )
                }
            }
        }
    }

    private fun refreshRecentCaptures() {
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            val recent = queryAppRecentCaptures(appContext)
            if (recent.isEmpty()) {
                if (!container.processing.state.value.active) {
                    _ui.update { it.copy(last = null, recentUris = emptyList()) }
                }
                return@launch
            }
            val (firstUri, firstMime) = recent.first()
            val thumb = container.output.thumbnail(firstUri)
            _ui.update { s ->
                val keepInMemory = s.last != null && s.last.uri == firstUri && s.last.thumbnail != null
                s.copy(
                    last = if (keepInMemory) s.last else LastCapture(firstUri, firstMime, thumb),
                    recentUris = recent.map { it.first },
                )
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
        refreshRecentCaptures()
    }

    fun onPause() {
        resumed = false
        shutterJob?.takeIf { _ui.value.busy == Busy.COUNTDOWN }?.cancel()
        _ui.update { it.copy(awaitingGallery = false) }
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
        _ui.update {
            it.copy(
                current = info,
                zoom = 1f,
                cropZoom = 1f,
                zoomTarget = null,
                zoomAnimId = it.zoomAnimId + 1,
                evIndex = 0,
                wbShift = 0,
                focus = null,
                focusHeld = false,
            )
        }
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

    /** Tap on the selected main lens: 1x and the cropped 2x in turn; the viewfinder animates there. */
    fun toggleTwoX() {
        val s = _ui.value
        val cam = s.current ?: return
        if (cam.facing != LensFacing.BACK || cam.relativeZoom != 1f || s.busy == Busy.CAPTURING) return
        val target = if (s.cropZoom > 1f) 1f else TWO_X.coerceAtMost(cam.zoomRange.upper)
        _ui.update { it.copy(cropZoom = target, zoomTarget = target, zoomAnimId = it.zoomAnimId + 1) }
    }

    /** One frame of the [CameraUiState.zoomTarget] animation. */
    fun setZoom(value: Float) {
        val cam = _ui.value.current ?: return
        val z = value.coerceIn(cam.zoomRange.lower, cam.zoomRange.upper)
        if (abs(z - _ui.value.zoom) < 0.001f) return
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

    fun setWb(shift: Int) {
        val i = shift.coerceIn(-WB_MAX_STEPS, WB_MAX_STEPS)
        if (i == _ui.value.wbShift) return
        _ui.update { it.copy(wbShift = i) }
        pushControls()
    }

    fun tapFocus(x: Float, y: Float, onRing: Boolean) {
        val s = _ui.value
        if (s.focus != null) {
            if (!onRing) {
                releaseFocus()
            } else if (!s.focusHeld) {
                focusResetJob?.cancel()
                _ui.update { it.copy(focusHeld = true) }
            }
            return
        }
        val cam = s.current ?: return
        val (u, v) = viewToSensor(x, y, cam.sensorOrientation, cam.facing == LensFacing.FRONT, s.zoom)
        if (s.evIndex != 0 || s.wbShift != 0) {
            _ui.update { it.copy(evIndex = 0, wbShift = 0) }
            pushControls()
        }
        controller.focusAt(u, v, s.zoom)
        _ui.update { it.copy(focus = FocusPoint(x, y, ++focusSeq), focusHeld = false) }
        focusResetJob?.cancel()
        focusResetJob = viewModelScope.launch {
            delay(FOCUS_HOLD_MS)
            releaseFocus()
        }
    }

    private fun releaseFocus() {
        focusResetJob?.cancel()
        if (_ui.value.focus == null) return
        _ui.update { it.copy(focus = null, focusHeld = false) }
        controller.resetFocus()
    }

    fun setFlash(v: FlashMode) = viewModelScope.launch {
        settingsRepo.setFlash(v)
        pushControls(v)
    }

    fun setTimer(v: TimerMode) = viewModelScope.launch { settingsRepo.setTimer(v) }
    fun setAspect(v: AspectRatio) = viewModelScope.launch { settingsRepo.setAspect(v) }
    fun setGrid(v: Boolean) = viewModelScope.launch { settingsRepo.setGrid(v) }

    fun onThumbnailClick() {
        val s = _ui.value
        val active = container.processing.state.value.active || s.busy == Busy.CAPTURING || s.busy == Busy.PROCESSING
        if (active) {
            _ui.update { it.copy(awaitingGallery = !it.awaitingGallery) }
        } else {
            _ui.update { it.copy(openGalleryPending = true, galleryOpenId = it.galleryOpenId + 1) }
        }
    }

    fun consumeGalleryOpen() {
        _ui.update { it.copy(openGalleryPending = false) }
    }

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
        val cfg = settings.value
        val liveNow = live.value.copy(motionRadPerSec = motion.rms())
        val fires = flashFires(cfg, cam, liveNow)
        val base = basePipelineFor(cfg, cam, fires)
        val resolvedExposure = base.exposureFor(ExposurePlan.Auto(s.evIndex), liveNow, cam)
        val pipeline: ImagePipeline = if (cfg.debugCapture) {
            DebugCapturePipeline(
                delegate = base,
                debugRoot = container.debugCapturesDir,
                packageName = container.packageName,
                versionName = BuildConfig.VERSION_NAME,
                aspect = cfg.aspect,
                exposurePlan = resolvedExposure,
                liveExposure = liveNow,
            )
        } else {
            base
        }
        _ui.update { it.copy(busy = Busy.CAPTURING, progress = 0f, shutterTick = it.shutterTick + 1) }
        sound.play(MediaActionSound.SHUTTER_CLICK)
        val spec = CaptureSpec(
            frameCount = pipeline.framesPerCapture,
            exposure = resolvedExposure,
            flash = if (fires) cfg.flash else FlashMode.OFF,
            deviceRotation = deviceRotation,
            zoomRatio = s.zoom,
            cropZoom = s.cropZoom,
        )
        val stream = pipeline.openStream(spec.exposure)
        val burst = try {
            controller.capture(spec, stream)
        } catch (e: Exception) {
            stream?.close()
            Log.w(TAG, "capture failed", e)
            releaseFocus()
            fail(CaptureMessage.CAPTURE_FAILED)
            return
        }
        releaseFocus()
        container.processing.enqueue(pipeline, burst)
        if (!controller.budget.canAfford(nextFrames())) viewModelScope.launch { controller.reopen() }
        _ui.update {
            val q = container.processing.state.value
            it.copy(
                busy = if (q.isFull) Busy.PROCESSING else Busy.IDLE,
                backgroundProcessing = q.active,
                progress = q.progress,
            )
        }
    }

    /** Single unadjustable Indigo / Pixel-1 pipeline; flash reduces burst to 1 frame. */
    private fun basePipelineFor(
        cfg: CameraSettings,
        cam: CameraInfo,
        fires: Boolean = flashFires(cfg, cam, live.value),
    ): ImagePipeline {
        val baseOptions = PipelineOptions(
            writeJpeg = true,
            writeDng = false,
            look = BurstLook(style = "pixel"),
            lca = LensCalibration.lca(cam.id),
            noise = LensCalibration.noise(cam.id),
            gpu = true,
            aspectRatio = if (cfg.aspect == AspectRatio.SIXTEEN_NINE) 16f / 9f else 0f,
        )
        val options = if (fires) baseOptions.copy(frames = 1) else baseOptions
        return PipelineRegistry.create(PipelineRegistry.defaultId, options)
    }

    private fun nextFrames(): Int = _ui.value.current?.let { basePipelineFor(settings.value, it).framesPerCapture } ?: 1

    private fun fail(message: CaptureMessage) {
        _ui.update {
            it.copy(
                busy = Busy.IDLE,
                countdown = null,
                awaitingGallery = false,
                message = message,
                messageId = it.messageId + 1,
            )
        }
    }

    private fun previewControls(flash: FlashMode = settings.value.flash): PreviewControls {
        val s = _ui.value
        return PreviewControls(
            exposure = ExposurePlan.Auto(s.evIndex),
            flash = flash,
            zoomRatio = s.zoom,
            wbShift = s.wbShift,
        )
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
        const val WB_MAX_STEPS = 10
        private const val TWO_X = 2f

        fun flashFires(cfg: CameraSettings, cam: CameraInfo, live: LiveExposure): Boolean =
            cam.hasFlash && (cfg.flash == FlashMode.ON || cfg.flash == FlashMode.AUTO && live.flashRequired)

        internal fun sensorPointToView(u: Float, v: Float, sensorOrientation: Int, mirrored: Boolean, zoom: Float = 1f): Pair<Float, Float> {
            val z = zoom.coerceAtLeast(1f)
            val uz = 0.5f + (u - 0.5f) * z
            val vz = 0.5f + (v - 0.5f) * z
            val (vx, y) = when (sensorOrientation) {
                90 -> 1f - vz to uz
                180 -> 1f - uz to 1f - vz
                270 -> vz to 1f - uz
                else -> uz to vz
            }
            return (if (mirrored) 1f - vx else vx) to y
        }

        fun sensorToView(r: RectF, sensorOrientation: Int, mirrored: Boolean, zoom: Float = 1f): RectF {
            val (x0, y0) = sensorPointToView(r.left, r.top, sensorOrientation, mirrored, zoom)
            val (x1, y1) = sensorPointToView(r.right, r.bottom, sensorOrientation, mirrored, zoom)
            return RectF(minOf(x0, x1), minOf(y0, y1), maxOf(x0, x1), maxOf(y0, y1))
        }

        fun viewToSensor(x: Float, y: Float, sensorOrientation: Int, mirrored: Boolean, zoom: Float = 1f): Pair<Float, Float> {
            val vx = if (mirrored) 1f - x else x
            val (uz, vz) = when (sensorOrientation) {
                90 -> y to 1f - vx
                180 -> 1f - vx to 1f - y
                270 -> 1f - y to vx
                else -> vx to y
            }
            val z = zoom.coerceAtLeast(1f)
            return (0.5f + (uz - 0.5f) / z).coerceIn(0f, 1f) to (0.5f + (vz - 0.5f) / z).coerceIn(0f, 1f)
        }
    }
}
