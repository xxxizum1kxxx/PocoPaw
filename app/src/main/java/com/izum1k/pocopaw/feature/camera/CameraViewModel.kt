package com.izum1k.pocopaw.feature.camera

import android.graphics.Bitmap
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.media.MediaActionSound
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.izum1k.pocopaw.AppContainer
import com.izum1k.pocopaw.camera.AwbEstimatorMode
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
import com.izum1k.pocopaw.data.DngContent
import com.izum1k.pocopaw.data.KrScaleMode
import com.izum1k.pocopaw.data.OutputFormat
import com.izum1k.pocopaw.data.Processing
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.data.SceneMode
import com.izum1k.pocopaw.data.BurstFrames
import com.izum1k.pocopaw.data.LookStyle
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import com.izum1k.pocopaw.pipeline.BurstExposure
import com.izum1k.pocopaw.pipeline.BurstLook
import com.izum1k.pocopaw.pipeline.BurstScene
import com.izum1k.pocopaw.pipeline.DngPassthroughPipeline
import com.izum1k.pocopaw.pipeline.FlowMethod
import com.izum1k.pocopaw.pipeline.LensCalibration
import com.izum1k.pocopaw.pipeline.HdrPlusPipeline
import com.izum1k.pocopaw.pipeline.ImagePipeline
import com.izum1k.pocopaw.pipeline.PipelineOptions
import com.izum1k.pocopaw.pipeline.PipelineRegistry
import com.izum1k.pocopaw.pipeline.TonemapMode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
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

data class LastCapture(val uri: Uri, val mimeType: String, val thumbnail: Bitmap?)

/** The last pipeline run, for the developer chip. */
data class PipelineRun(val pipelineId: String, val frames: Int, val millis: Long)

/** The quick settings panel's choices; null keeps the saved default. */
data class SessionSettings(
    val output: OutputFormat? = null,
    val processing: Processing? = null,
    val krScale: KrScaleMode? = null,
    val burstFrames: BurstFrames? = null,
    val scene: SceneMode? = null,
    val look: LookStyle? = null,
    val tonemapMode: TonemapMode? = null,
    val ultraHdr: Boolean? = null,
    val earlyStopEnabled: Boolean? = null,
) {
    fun over(saved: CameraSettings) = saved.copy(
        output = output ?: saved.output,
        processing = processing ?: saved.processing,
        krScale = krScale ?: saved.krScale,
        burstFrames = burstFrames ?: saved.burstFrames,
        scene = scene ?: saved.scene,
        look = look ?: saved.look,
        tonemapMode = tonemapMode ?: saved.tonemapMode,
        ultraHdr = ultraHdr ?: saved.ultraHdr,
        earlyStopEnabled = earlyStopEnabled ?: saved.earlyStopEnabled,
    )

    /** Drops the choices whose saved default just changed. */
    fun without(prev: CameraSettings, saved: CameraSettings) = copy(
        output = output.takeIf { prev.output == saved.output },
        processing = processing.takeIf { prev.processing == saved.processing },
        krScale = krScale.takeIf { prev.krScale == saved.krScale },
        burstFrames = burstFrames.takeIf { prev.burstFrames == saved.burstFrames },
        scene = scene.takeIf { prev.scene == saved.scene },
        look = look.takeIf { prev.look == saved.look },
        tonemapMode = tonemapMode.takeIf { prev.tonemapMode == saved.tonemapMode },
        ultraHdr = ultraHdr.takeIf { prev.ultraHdr == saved.ultraHdr },
        earlyStopEnabled = earlyStopEnabled.takeIf { prev.earlyStopEnabled == saved.earlyStopEnabled },
    )
}

data class CameraUiState(
    val cameras: List<CameraInfo> = emptyList(),
    val current: CameraInfo? = null,
    val mode: CaptureMode = CaptureMode.PHOTO,
    val zoom: Float = 1f,
    /** The main lens's 2x step: the photo is cropped to the centre 1/[cropZoom]. */
    val cropZoom: Float = 1f,
    /** Where the viewfinder animates [zoom] to (null: no animation); [zoomAnimId] restarts or cancels it. */
    val zoomTarget: Float? = null,
    val zoomAnimId: Int = 0,
    val evIndex: Int = 0,
    /** Warm (+) / cool (−) steps off auto white balance, within ±[WB_MAX_STEPS]. */
    val wbShift: Int = 0,
    val manualIso: Int = 100,
    val manualExposureNs: Long = 10_000_000L,
    /** Pro: AE picks the ISO (with a manual shutter, the one that keeps AE's exposure). */
    val isoAuto: Boolean = true,
    /** Pro: AE picks the shutter (with a manual ISO, the one that keeps AE's exposure). */
    val shutterAuto: Boolean = true,
    val focus: FocusPoint? = null,
    /** The tap's AF/AE lock holds until released, instead of for [FOCUS_HOLD_MS]. */
    val focusHeld: Boolean = false,
    val countdown: Int? = null,
    val busy: Busy = Busy.IDLE,
    val backgroundProcessing: Boolean = false,
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

    /** Quick settings changed in the viewfinder: this session's, over the saved defaults, never saved. */
    private val session = MutableStateFlow(SessionSettings())

    val settings: StateFlow<CameraSettings> = combine(settingsRepo.settings, session) { saved, s -> s.over(saved) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, session.value.over(settingsRepo.settings.value))
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
        // A default changed in Settings wins over this session's choice for that setting.
        viewModelScope.launch {
            var prev = settingsRepo.settings.value
            settingsRepo.settings.collect { saved ->
                session.update { it.without(prev, saved) }
                prev = saved
            }
        }
        viewModelScope.launch {
            settings.collect { cfg ->
                if (needsMotion(cfg)) motion.start() else motion.stop()
                controller.setRaw10Enabled(cfg.raw10Enabled)
                pushControls(cfg.flash)
            }
        }
        // A setting that asks for more frames than are left (8 to 16, DNG to a burst, 2x zoom boost)
        // reopens right away instead of at the next press.
        viewModelScope.launch {
            combine(
                settings,
                cameraState,
                _ui.map { it.cropZoom }.distinctUntilChanged(),
            ) { _, state, _ -> state }.collect { state ->
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
            container.output.latest()?.let { latest ->
                if (_ui.value.last == null) _ui.update { it.copy(last = LastCapture(latest.uri, latest.mimeType, container.output.thumbnail(latest.uri))) }
            }
        }
        viewModelScope.launch {
            var seenErrors = container.processing.state.value.errorSeq
            container.processing.state.collect { st ->
                if (st.errorSeq > seenErrors) {
                    seenErrors = st.errorSeq
                    _ui.update { it.copy(message = CaptureMessage.SAVE_FAILED, messageId = it.messageId + 1) }
                }
                _ui.update { ui ->
                    ui.copy(
                        backgroundProcessing = st.active,
                        progress = st.progress,
                        busy = if (ui.busy == Busy.PROCESSING && !st.isFull) Busy.IDLE else ui.busy,
                        last = st.last?.let { LastCapture(it.uri, it.mimeType, it.thumbnail) } ?: ui.last,
                        lastRun = st.lastRun?.let { PipelineRun(it.pipelineId, it.frames, it.millis) } ?: ui.lastRun,
                    )
                }
            }
        }
    }

    fun onPermission(granted: Boolean) {
        permitted = granted
        maybeOpen()
    }

    fun onResume() {
        resumed = true
        container.syncAwbEstimator()
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
        _ui.update {
            it.copy(
                current = info, zoom = 1f, cropZoom = 1f, zoomTarget = null, zoomAnimId = it.zoomAnimId + 1,
                evIndex = 0, wbShift = 0, focus = null, focusHeld = false,
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

    /** Tap on the selected main lens: 1x and the cropped 2x in turn; the viewfinder animates there. */
    fun toggleTwoX() {
        val s = _ui.value
        val cam = s.current ?: return
        if (cam.facing != LensFacing.BACK || cam.relativeZoom != 1f || s.busy == Busy.CAPTURING) return
        val target = if (s.cropZoom > 1.01f) 1f else TWO_X.coerceAtMost(cam.zoomRange.upper).coerceAtLeast(cam.zoomRange.lower)
        _ui.update { it.copy(cropZoom = target, zoomTarget = target, zoomAnimId = it.zoomAnimId + 1) }
    }

    /** One frame of the [CameraUiState.zoomTarget] animation, or continuous zoom update. */
    fun setZoom(value: Float) {
        val cam = _ui.value.current ?: return
        val z = value.coerceIn(cam.zoomRange.lower, cam.zoomRange.upper)
        if (abs(z - _ui.value.zoom) < 0.001f) return
        _ui.update {
            it.copy(
                zoom = z,
                cropZoom = if (it.zoomTarget == null) cropZoomFor(cam, z) else it.cropZoom,
            )
        }
        pushControls()
    }

    /** Continuous pinch zoom (`1.0x..2.0x` on main back lens, clamped to `cam.zoomRange`). */
    fun setContinuousZoom(value: Float) {
        if (!value.isFinite()) return
        val cam = _ui.value.current ?: return
        val isMainBack = cam.facing == LensFacing.BACK && cam.relativeZoom == 1f
        val z = clampContinuousZoom(value, isMainBack, cam.zoomRange.lower, cam.zoomRange.upper, fallback = _ui.value.zoom)
        if (abs(z - _ui.value.zoom) < 0.001f && _ui.value.zoomTarget == null) return
        _ui.update {
            it.copy(
                zoom = z,
                cropZoom = cropZoomFor(cam, z),
                zoomTarget = null,
                zoomAnimId = if (it.zoomTarget != null) it.zoomAnimId + 1 else it.zoomAnimId,
            )
        }
        pushControls()
    }

    /** Multiplies the latest continuous zoom by a pinch [factor] without reading stale Compose state. */
    fun scaleContinuousZoom(factor: Float) {
        if (!factor.isFinite() || factor <= 0f) return
        setContinuousZoom(_ui.value.zoom * factor)
    }

    private fun cropZoomFor(cam: CameraInfo?, zoom: Float): Float =
        cropZoomFor(
            isMainBack = cam != null && cam.facing == LensFacing.BACK && cam.relativeZoom == 1f,
            zoom = zoom,
            zoomUpper = cam?.zoomRange?.upper ?: 1f,
        )

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

    /**
     * A tap at normalised texture coordinates (portrait, as displayed). With nothing locked it locks
     * AF and AE at that point for [FOCUS_HOLD_MS], with EV and white balance back at 0; a tap on that
     * lock's ring holds it until a photo is taken; a tap anywhere else releases it.
     */
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

    /** Back to continuous AF and AE, if a tap lock is on. */
    fun unlockFocus() = releaseFocus()

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

    /** A capture setting changed from the quick settings panel, for this session only. */
    fun setQuick(change: SessionSettings.() -> SessionSettings) = session.update(change)
    fun setAwbMode(v: AwbEstimatorMode) = viewModelScope.launch { settingsRepo.setAwbMode(v) }
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
        val cfg = settings.value
        val fires = flashFires(cfg, cam, live.value)
        val shake = motion.rms()
        // KR's super-resolution comes from hand shake; on a tripod it only costs time
        val still = cfg.processing == Processing.HDR_PLUS_KR && shake != null && shake < STILL_RAD_PER_SEC
        if (still) Log.i(TAG, "HDR+ KR -> HDR+: still (gyro rms %.4f rad/s)".format(shake))
        _ui.update { it.copy(busy = Busy.CAPTURING, progress = 0f) }
        val pipeline = try {
            withContext(Dispatchers.IO) { pipelineFor(cfg, cam, fires, still, s.cropZoom) }
        } catch (e: CancellationException) {
            _ui.update { it.copy(busy = Busy.IDLE) }
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "pipeline setup failed", e)
            releaseFocus()
            fail(CaptureMessage.CAPTURE_FAILED)
            return
        }
        _ui.update { it.copy(shutterTick = it.shutterTick + 1) }
        if (cfg.shutterSound) sound.play(MediaActionSound.SHUTTER_CLICK)
        val zslPolicy = zslCapturePolicy(
            cfg = cfg,
            mode = s.mode,
            flash = fires,
            frames = pipeline.framesPerCapture,
            clippedFraction = live.value.clippedFraction,
            bufferedZsl = controller.bufferedZslFrames,
        )
        val spec = CaptureSpec(
            frameCount = pipeline.framesPerCapture,
            exposure = pipeline.exposureFor(captureExposure(s, cam, fires), live.value.copy(motionRadPerSec = shake), cam),
            flash = if (fires) cfg.flash else FlashMode.OFF,
            deviceRotation = deviceRotation,
            zoomRatio = s.zoom,
            cropZoom = s.cropZoom,
            earlyStopEnabled = cfg.isEarlyStopEnabled() && cfg.earlyStopEnabled,
            allowZsl = zslPolicy.allowZsl,
            reserveShortFrames = zslPolicy.reserveShortFrames,
        )
        val stream = if (zslPolicy.disableStreamForHybrid) null else pipeline.openStream(spec.exposure)
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
        // The burst is in memory now; reopening while it merges keeps the next press from waiting.
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

    /**
     * DNG output keeps the single-frame pass-through; JPEG output runs the burst pipeline. A flash
     * fires for one frame only, so a flash capture is a burst of one. [still] runs KR as plain HDR+.
     */
    internal fun pipelineFor(
        cfg: CameraSettings,
        cam: CameraInfo,
        flash: Boolean = flashFires(cfg, cam, live.value),
        still: Boolean = false,
        cropZoom: Float = _ui.value.cropZoom,
    ): ImagePipeline {
        if (cfg.output == OutputFormat.DNG) return PipelineRegistry.create(DngPassthroughPipeline.ID)
        return PipelineRegistry.create(
            pipelineIdFor(cfg, still),
            buildPipelineOptions(
                cfg = cfg,
                camId = cam.id,
                flash = flash,
                cropZoom = cropZoom,
                lutPathResolver = container::lutPathFor,
                pmridPath = container::pmridModelPath,
                flowPath = container::seaRaftModelPath,
                hdrnetPath = container::hdrnetModelPath,
                segPath = container::segModelPath,
            ),
        )
    }

    /** RAW frames the next press will ask for, computed without resolving asset paths. */
    private fun nextFrames(): Int {
        val s = _ui.value
        val cam = s.current ?: return 1
        val cfg = settings.value
        return expectedFrames(cfg, flashFires(cfg, cam, live.value), s.cropZoom)
    }

    private fun fail(message: CaptureMessage) {
        _ui.update { it.copy(busy = Busy.IDLE, countdown = null, message = message, messageId = it.messageId + 1) }
    }

    /** What the preview runs: Camera2 has no half-auto, so unless both are manual AE keeps metering. */
    private fun exposurePlan(s: CameraUiState): ExposurePlan = when {
        s.mode == CaptureMode.PHOTO -> ExposurePlan.Auto(s.evIndex)
        s.isoAuto || s.shutterAuto -> ExposurePlan.Auto()
        else -> ExposurePlan.Manual(ManualExposure(s.manualIso, s.manualExposureNs))
    }

    /**
     * With the flash firing, anything left on auto goes to AE's precapture flash metering: the
     * preview metered the scene without the flash, so a priority exposure from it plus a full
     * flash would overexpose. Fully manual keeps the user's exposure, with the flash at full power.
     */
    private fun captureExposure(s: CameraUiState, cam: CameraInfo, flash: Boolean): ExposurePlan {
        if (flash && s.mode == CaptureMode.PRO && (s.isoAuto || s.shutterAuto)) return ExposurePlan.Auto()
        if (s.mode != CaptureMode.PRO || s.isoAuto == s.shutterAuto) return exposurePlan(s)
        return priorityExposure(s, live.value, cam)?.let { ExposurePlan.Manual(it) } ?: ExposurePlan.Auto()
    }

    private fun previewControls(flash: FlashMode = settings.value.flash): PreviewControls {
        val s = _ui.value
        val cfg = settings.value
        val zslEligible = cfg.zslEnabled && s.mode == CaptureMode.PHOTO && cfg.scene != SceneMode.NIGHT && flash != FlashMode.ON
        return PreviewControls(exposure = exposurePlan(s), flash = flash, zoomRatio = s.zoom, wbShift = s.wbShift, zslActive = zslEligible)
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
        /** Gyro RMS below which the phone is on a support: ~0.001 on a tripod, above 0.02 in hand. */
        private const val STILL_RAD_PER_SEC = 0.005f

        /** Night meters hand shake for its exposure; KR checks it to skip super-resolution when still. */
        fun needsMotion(cfg: CameraSettings) = cfg.scene == SceneMode.NIGHT || cfg.processing == Processing.HDR_PLUS_KR

        internal data class ZslCapturePolicy(
            val allowZsl: Boolean,
            val reserveShortFrames: Int,
            val disableStreamForHybrid: Boolean,
        )

        internal fun zslCapturePolicy(
            cfg: CameraSettings,
            mode: CaptureMode,
            flash: Boolean,
            frames: Int,
            clippedFraction: Float,
            bufferedZsl: Int,
        ): ZslCapturePolicy {
            val allowZsl = cfg.zslEnabled && mode == CaptureMode.PHOTO && cfg.scene != SceneMode.NIGHT && !flash && frames > 1
            val reserveShortFrames = if (allowZsl && cfg.scene == SceneMode.AUTO && clippedFraction > BurstExposure.HIGHLIGHT_CLIP_SAFE) {
                minOf(2, frames - 1)
            } else {
                0
            }
            val earlyStopActive = cfg.isEarlyStopEnabled() && cfg.earlyStopEnabled
            val disableStreamForHybrid = allowZsl && bufferedZsl > 0 && !earlyStopActive
            return ZslCapturePolicy(
                allowZsl = allowZsl,
                reserveShortFrames = reserveShortFrames,
                disableStreamForHybrid = disableStreamForHybrid,
            )
        }

        internal fun pipelineIdFor(cfg: CameraSettings, still: Boolean = false): String = when {
            cfg.output == OutputFormat.DNG -> DngPassthroughPipeline.ID
            cfg.processing == Processing.HDR_PLUS -> HdrPlusPipeline.ID
            still -> HdrPlusPipeline.ID
            else -> HdrPlusPipeline.ID_KR
        }

        internal fun effectiveLook(
            cfg: CameraSettings,
            flash: Boolean = false,
            lutPathResolver: (String) -> String = { "" },
        ): BurstLook {
            val nightLook = !flash && cfg.scene == SceneMode.NIGHT
            val lutId = if (nightLook) null else cfg.look.lutId
            val resolvedLut = lutId?.let(lutPathResolver).orEmpty()
            return BurstLook(
                style = if (nightLook) "night" else cfg.look.preset,
                lutPath = resolvedLut,
                lutStrength = if (lutId != null && resolvedLut.isNotEmpty()) 1f else 0f,
            )
        }

        internal fun expectedFrames(
            cfg: CameraSettings,
            flash: Boolean = false,
            cropZoom: Float = 1f,
        ): Int = if (cfg.output == OutputFormat.DNG) {
            1
        } else {
            BurstExposure.framesForZoom(cfg.burstFrames.count, cropZoom, flash)
        }

        internal fun buildPipelineOptions(
            cfg: CameraSettings,
            camId: String,
            flash: Boolean = false,
            cropZoom: Float = 1f,
            lutPathResolver: (String) -> String = { "" },
            pmridPath: () -> String = { "" },
            flowPath: () -> String = { "" },
            hdrnetPath: () -> String = { "" },
            segPath: () -> String = { "" },
        ): PipelineOptions {
            val boostFrames = expectedFrames(cfg, flash, cropZoom)
            val pmridActive = cfg.pmridEnabled && cfg.processing == Processing.HDR_PLUS
            val flowActive = cfg.flowMethod != FlowMethod.BLOCK_MATCH
            val hdrnetActive = cfg.tonemapMode == TonemapMode.HDRNET
            val segActive = cfg.semanticSegEnabled
            return PipelineOptions(
                frames = boostFrames,
                writeJpeg = true,
                writeDng = cfg.output == OutputFormat.DNG_JPEG,
                mergedDng = cfg.effectiveDngContent() == DngContent.MERGED,
                look = effectiveLook(cfg, flash, lutPathResolver),
                scene = if (flash) BurstScene.AUTO else when (cfg.scene) {
                    SceneMode.AUTO -> BurstScene.AUTO
                    SceneMode.BRACKET -> BurstScene.BRACKET
                    SceneMode.NIGHT -> BurstScene.NIGHT
                },
                ultraHdr = cfg.ultraHdr,
                lca = LensCalibration.lca(camId),
                noise = LensCalibration.noise(camId),
                gpu = cfg.effectiveGpuAcceleration(),
                aspectRatio = if (cfg.aspect == AspectRatio.SIXTEEN_NINE) 16f / 9f else 0f,
                demosaic = cfg.demosaicMethod,
                krScale = cfg.krScale.scale,
                krAuto2x = cfg.krScale.auto2x,
                pmridEnabled = pmridActive,
                pmridModelPath = if (pmridActive) pmridPath() else "",
                accelBackend = cfg.accelBackend,
                flowMethod = cfg.flowMethod,
                flowModelPath = if (flowActive) flowPath() else "",
                tonemapMode = cfg.tonemapMode,
                hdrnetModelPath = if (hdrnetActive) hdrnetPath() else "",
                semanticSegEnabled = segActive,
                segModelPath = if (segActive) segPath() else "",
                earlyStopEnabled = cfg.isEarlyStopEnabled() && cfg.earlyStopEnabled,
            )
        }
        const val WB_MAX_STEPS = 10
        private const val TWO_X = 2f

        internal fun clampContinuousZoom(
            value: Float,
            isMainBack: Boolean,
            zoomLower: Float,
            zoomUpper: Float,
            fallback: Float = 1f,
        ): Float {
            if (!value.isFinite()) return fallback
            val lo = minOf(zoomLower, zoomUpper)
            val hi = maxOf(zoomLower, zoomUpper)
            val minZ = lo.coerceAtLeast(1f)
            val maxZ = if (isMainBack) {
                TWO_X.coerceAtMost(hi).coerceAtLeast(lo)
            } else {
                hi
            }.coerceAtLeast(minZ)
            return value.coerceIn(minZ, maxZ)
        }

        internal fun cropZoomFor(
            isMainBack: Boolean,
            zoom: Float,
            zoomUpper: Float,
        ): Float =
            if (isMainBack && zoom.isFinite()) {
                zoom.coerceIn(1f, TWO_X.coerceAtMost(zoomUpper).coerceAtLeast(1f))
            } else {
                1f
            }

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

        /** On always fires; Auto only when the preview AE asks for it, so a lit scene keeps its burst. */
        fun flashFires(cfg: CameraSettings, cam: CameraInfo, live: LiveExposure): Boolean =
            cam.hasFlash && (cfg.flash == FlashMode.ON || cfg.flash == FlashMode.AUTO && live.flashRequired)

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
         * Active-array normalised point ([0..1]) to displayed (portrait, mirrored for the front camera),
         * accounting for preview [zoom] around the center of the active array.
         */
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

        /** [viewToSensor] undone, for a rectangle: what the face detector reports, as displayed. */
        fun sensorToView(r: RectF, sensorOrientation: Int, mirrored: Boolean, zoom: Float = 1f): RectF {
            val (x0, y0) = sensorPointToView(r.left, r.top, sensorOrientation, mirrored, zoom)
            val (x1, y1) = sensorPointToView(r.right, r.bottom, sensorOrientation, mirrored, zoom)
            return RectF(minOf(x0, x1), minOf(y0, y1), maxOf(x0, x1), maxOf(y0, y1))
        }

        /**
         * Displayed (portrait, mirrored for the front camera) to active-array sensor coordinates.
         * The preview shows the sensor rotated clockwise by its orientation and cropped by [zoom].
         */
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
