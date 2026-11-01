package com.izum1k.pocopaw.camera

import android.annotation.SuppressLint
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.Face
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.pipeline.NativeMemory
import com.izum1k.pocopaw.pipeline.RawBurst
import com.izum1k.pocopaw.pipeline.RawFrame
import com.izum1k.pocopaw.pipeline.RawMetadata
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.android.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import java.nio.ByteBuffer
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class CameraFailure(val reason: CameraController.Reason, cause: Throwable? = null) : Exception(reason.name, cause)

/**
 * The only owner of a CameraDevice. Preview requests carry the YUV preview surface only; capture
 * requests add the RAW surface and are counted against [budget]. Every Camera2 call runs on one
 * handler thread, serialised by a mutex.
 */
class CameraController(private val manager: CameraManager) {

    sealed interface State {
        data object Closed : State
        data class Opening(val cameraId: String) : State
        data class Previewing(val cameraId: String) : State
        data class Capturing(val cameraId: String) : State
        data class Error(val cameraId: String?, val reason: Reason) : State
    }

    enum class Reason { IN_USE, MAX_IN_USE, DISABLED, DEVICE, SERVICE, DISCONNECTED, CONFIGURATION, TIMEOUT, UNKNOWN }

    private val thread = HandlerThread("PocoPawCamera").apply { start() }
    private val handler = Handler(thread.looper)
    private val executor = Executor { handler.post(it) }
    private val dispatcher = handler.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    private val mutex = Mutex()

    val budget = RawRequestBudget()

    private val _state = MutableStateFlow<State>(State.Closed)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _live = MutableStateFlow(LiveExposure())
    val live: StateFlow<LiveExposure> = _live.asStateFlow()

    private var camera: CameraInfo? = null
    private var texture: SurfaceTexture? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var rawReader: ImageReader? = null
    private var previewSurface: Surface? = null
    private var controls = PreviewControls()
    private var focusRegion: MeteringRectangle? = null
    /** AE held at the tap's metering; set once AE has settled on the tap region. */
    private var aeLock = false
    private var aeLockAfter = 0L
    private var aeLockDeadline = 0L
    /** The largest face, as a 3A region; a tap's [focusRegion] overrides it. */
    private var faceRegion: MeteringRectangle? = null
    private var faceSeenAt = 0L
    private var faceRegionAt = 0L
    private var activeCapture: CaptureCollector? = null
    private var aeWatcher: ((Int?) -> Unit)? = null
    private var afWatcher: ((Int?) -> Unit)? = null
    private var recoveries = 0

    suspend fun open(info: CameraInfo, texture: SurfaceTexture, controls: PreviewControls) = withContext(dispatcher) {
        mutex.withLock {
            this@CameraController.controls = controls
            focusRegion = null
            aeLock = false
            aeLockDeadline = 0L
            openLocked(info, texture)
        }
    }

    suspend fun close() = withContext(dispatcher) {
        mutex.withLock {
            closeLocked()
            camera = null
            texture = null
            _state.value = State.Closed
        }
    }

    /** Reopens the last camera after an error the UI offered to retry. */
    suspend fun retry() = withContext(dispatcher) {
        mutex.withLock {
            val info = camera ?: return@withLock
            val tex = texture ?: return@withLock
            recoveries = 0
            openLocked(info, tex)
        }
    }

    /** Closes and reopens the current camera, which gives back the whole RAW budget. */
    suspend fun reopen() = withContext(dispatcher) {
        mutex.withLock {
            val info = camera ?: return@withLock
            val tex = texture ?: return@withLock
            if (_state.value !is State.Previewing) return@withLock
            Log.i(TAG, "reopening ${info.id} on request, ${budget.remaining.value} RAW requests left")
            openLocked(info, tex)
        }
    }

    fun updateControls(next: PreviewControls) {
        scope.launch {
            mutex.withLock {
                controls = next
                if (_state.value is State.Previewing) startPreview()
            }
        }
    }

    /**
     * Focuses once at a point in normalised sensor coordinates (0..1, sensor's own orientation) and
     * holds it, meters there and locks AE once it has settled, until [resetFocus].
     */
    fun focusAt(u: Float, v: Float) {
        scope.launch {
            mutex.withLock {
                val info = camera ?: return@withLock
                val s = session ?: return@withLock
                if (_state.value !is State.Previewing) return@withLock
                focusRegion = meteringRect(info.activeArray, u, v)
                aeLock = false
                val now = System.currentTimeMillis()
                aeLockAfter = now + AE_LOCK_MIN_MS
                aeLockDeadline = now + AE_LOCK_MAX_MS
                runCatching {
                    val cancel = previewBuilder() ?: return@withLock
                    cancel.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)
                    s.capture(cancel.build(), null, handler)
                    startPreview()
                    val trigger = previewBuilder() ?: return@withLock
                    trigger.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
                    s.capture(trigger.build(), null, handler)
                }.onFailure { Log.w(TAG, "focus failed", it) }
            }
        }
    }

    /** Back to continuous focus, whole-frame metering and running AE. */
    fun resetFocus() {
        scope.launch {
            mutex.withLock {
                if (focusRegion == null && !aeLock) return@withLock
                focusRegion = null
                aeLock = false
                aeLockDeadline = 0L
                val s = session ?: return@withLock
                if (_state.value !is State.Previewing) return@withLock
                runCatching {
                    val cancel = previewBuilder() ?: return@withLock
                    cancel.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)
                    s.capture(cancel.build(), null, handler)
                    startPreview()
                }
            }
        }
    }

    /** Captures [CaptureSpec.frameCount] RAW frames, reopening the device first if the budget cannot cover them. */
    suspend fun capture(spec: CaptureSpec): RawBurst = withContext(dispatcher) {
        mutex.withLock {
            var info = camera ?: throw CameraFailure(Reason.UNKNOWN)
            if (!budget.canAfford(spec.frameCount)) {
                Log.i(TAG, "RAW budget at ${budget.remaining.value}, reopening ${info.id}")
                openLocked(info, texture ?: throw CameraFailure(Reason.UNKNOWN))
                info = camera ?: throw CameraFailure(Reason.UNKNOWN)
            }
            val s = session ?: throw CameraFailure(Reason.UNKNOWN)
            val reader = rawReader ?: throw CameraFailure(Reason.UNKNOWN)
            _state.value = State.Capturing(info.id)
            var unlockFocus = false
            try {
                unlockFocus = lockFocus(s, info)
                val fires = info.hasFlash && spec.flash != FlashMode.OFF && spec.exposure is ExposurePlan.Auto
                if (fires) runPrecapture(s)
                val requests = List(spec.frameCount) { stillRequest(info, spec, reader.surface, it) }
                val collector = CaptureCollector(spec.frameCount)
                activeCapture = collector
                generateSequence { reader.acquireNextImage() }.forEach { it.close() }
                reader.setOnImageAvailableListener({ r -> r.acquireNextImage()?.let(collector::onImage) }, handler)
                budget.consume(spec.frameCount)
                s.captureBurst(requests, collector.callback, handler)
                val frames = withTimeout(captureTimeoutMs(spec)) { collector.await() }
                frames.first().result.let { r ->
                    Log.i(
                        TAG,
                        "captured ${frames.size}: ISO ${r.get(CaptureResult.SENSOR_SENSITIVITY)}, " +
                            "${r.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.div(1000)}us, flash state ${r.get(CaptureResult.FLASH_STATE)}, " +
                            "AE mode ${r.get(CaptureResult.CONTROL_AE_MODE)}, preview was ISO ${_live.value.iso} ${_live.value.exposureNs?.div(1000)}us",
                    )
                }
                RawBurst(
                    frames = frames,
                    metadata = RawMetadata.from(
                        cameraId = info.id,
                        facing = info.facing,
                        characteristics = info.characteristics,
                        cfa = info.cfa,
                        result = frames.first().result,
                        orientationDegrees = orientationFor(info, spec.deviceRotation),
                        zoomRatio = spec.zoomRatio,
                    ),
                )
            } catch (e: kotlinx.coroutines.TimeoutCancellationException) {
                throw CameraFailure(Reason.TIMEOUT, e)
            } finally {
                activeCapture?.release()
                activeCapture = null
                reader.setOnImageAvailableListener(null, null)
                if (device != null && session === s) {
                    _state.value = State.Previewing(info.id)
                    runCatching {
                        if (unlockFocus) previewBuilder()?.let { b ->
                            b.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)
                            s.capture(b.build(), null, handler)
                        }
                        startPreview()
                    }
                }
            }
        }
    }

    private suspend fun openLocked(info: CameraInfo, tex: SurfaceTexture) {
        closeLocked()
        camera = info
        texture = tex
        _state.value = State.Opening(info.id)
        try {
            tex.setDefaultBufferSize(info.previewSize.width, info.previewSize.height)
            val surface = Surface(tex).also { previewSurface = it }
            val reader = ImageReader.newInstance(info.rawSize.width, info.rawSize.height, ImageFormat.RAW_SENSOR, RAW_MAX_IMAGES)
            rawReader = reader
            val dev = openDevice(info.id)
            device = dev
            budget.reset()
            session = createSession(dev, listOf(surface, reader.surface))
            startPreview()
            _state.value = State.Previewing(info.id)
        } catch (e: CameraFailure) {
            Log.w(TAG, "open ${info.id} failed: ${e.reason}", e)
            closeLocked()
            _state.value = State.Error(info.id, e.reason)
        } catch (e: CameraAccessException) {
            Log.w(TAG, "open ${info.id} failed", e)
            closeLocked()
            _state.value = State.Error(info.id, reasonOfAccess(e.reason))
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "open ${info.id} failed", e)
            closeLocked()
            _state.value = State.Error(info.id, Reason.CONFIGURATION)
        } catch (e: SecurityException) {
            closeLocked()
            _state.value = State.Error(info.id, Reason.DISABLED)
        }
    }

    private fun closeLocked() {
        activeCapture?.fail(CameraFailure(Reason.DISCONNECTED))
        runCatching { session?.close() }
        runCatching { device?.close() }
        runCatching { rawReader?.close() }
        runCatching { previewSurface?.release() }
        session = null
        device = null
        rawReader = null
        previewSurface = null
        faceRegion = null
        _live.value = LiveExposure()
    }

    @SuppressLint("MissingPermission")
    private suspend fun openDevice(id: String): CameraDevice = suspendCancellableCoroutine { cont ->
        manager.openCamera(
            id,
            executor,
            object : CameraDevice.StateCallback() {
                override fun onOpened(d: CameraDevice) {
                    if (cont.isActive) cont.resume(d) else d.close()
                }

                override fun onDisconnected(d: CameraDevice) {
                    if (cont.isActive) cont.resumeWithException(CameraFailure(Reason.DISCONNECTED)) else onDeviceLost(d, Reason.DISCONNECTED)
                }

                override fun onError(d: CameraDevice, error: Int) {
                    val reason = reasonOfError(error)
                    if (cont.isActive) cont.resumeWithException(CameraFailure(reason)) else onDeviceLost(d, reason)
                }
            },
        )
    }

    private suspend fun createSession(d: CameraDevice, surfaces: List<Surface>): CameraCaptureSession =
        suspendCancellableCoroutine { cont ->
            val config = SessionConfiguration(
                SessionConfiguration.SESSION_REGULAR,
                surfaces.map { OutputConfiguration(it) },
                executor,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(s: CameraCaptureSession) {
                        if (cont.isActive) cont.resume(s) else s.close()
                    }

                    override fun onConfigureFailed(s: CameraCaptureSession) {
                        if (cont.isActive) cont.resumeWithException(CameraFailure(Reason.CONFIGURATION))
                    }
                },
            )
            d.createCaptureSession(config)
        }

    /**
     * ERROR_CAMERA_DEVICE (marble's RAW node leak among others) and provider restarts are
     * recovered by reopening, a few times, never by giving up the screen.
     */
    private fun onDeviceLost(d: CameraDevice, reason: Reason) {
        Log.w(TAG, "device ${d.id} lost: $reason, ${budget.remaining.value} RAW requests left")
        scope.launch {
            activeCapture?.fail(CameraFailure(reason))
            mutex.withLock {
                if (d !== device) return@withLock
                val info = camera
                val tex = texture
                closeLocked()
                val recoverable = reason == Reason.DEVICE || reason == Reason.SERVICE || reason == Reason.DISCONNECTED
                if (info == null || tex == null || !recoverable || recoveries >= MAX_RECOVERIES) {
                    _state.value = State.Error(info?.id, reason)
                    return@withLock
                }
                recoveries++
                _state.value = State.Opening(info.id)
                delay(if (reason == Reason.SERVICE) SERVICE_RESTART_DELAY_MS else 300L)
                if (camera === info && texture === tex) openLocked(info, tex)
            }
        }
    }

    private fun startPreview() {
        val s = session ?: return
        val b = previewBuilder() ?: return
        s.setRepeatingRequest(b.build(), previewCallback, handler)
    }

    private fun previewBuilder(): CaptureRequest.Builder? {
        val d = device ?: return null
        val info = camera ?: return null
        val surface = previewSurface ?: return null
        return d.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(surface)
            applyCommon(this, info, controls.exposure, controls.flash, controls.zoomRatio, preview = true)
        }
    }

    private fun stillRequest(info: CameraInfo, spec: CaptureSpec, raw: Surface, frame: Int): CaptureRequest {
        val d = device ?: throw CameraFailure(Reason.UNKNOWN)
        return d.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            previewSurface?.let(::addTarget)
            addTarget(raw)
            val exposure = (spec.exposure as? ExposurePlan.PerFrame)?.let { ExposurePlan.Manual(it.at(frame)) } ?: spec.exposure
            applyCommon(this, info, exposure, spec.flash, spec.zoomRatio, preview = false)
            set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CameraMetadata.STATISTICS_LENS_SHADING_MAP_MODE_ON)
        }.build()
    }

    private fun applyCommon(
        b: CaptureRequest.Builder,
        info: CameraInfo,
        exposure: ExposurePlan,
        flash: FlashMode,
        zoom: Float,
        preview: Boolean,
    ) {
        val c = info.characteristics
        b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        b.set(CaptureRequest.STATISTICS_FACE_DETECT_MODE, faceDetectMode(info))
        val tap = focusRegion
        val region = tap ?: faceRegion
        if (info.hasAutoFocus) {
            b.set(CaptureRequest.CONTROL_AF_MODE, if (tap != null) CameraMetadata.CONTROL_AF_MODE_AUTO else CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            if (region != null && (c.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0) > 0) b.set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(region))
        } else {
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
        }
        if (region != null && (c.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0) > 0) {
            b.set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(region))
        }
        when (exposure) {
            is ExposurePlan.Auto -> {
                val aeMode = when {
                    !info.hasFlash || flash == FlashMode.OFF -> CameraMetadata.CONTROL_AE_MODE_ON
                    flash == FlashMode.AUTO -> CameraMetadata.CONTROL_AE_MODE_ON_AUTO_FLASH
                    else -> CameraMetadata.CONTROL_AE_MODE_ON_ALWAYS_FLASH
                }
                b.set(CaptureRequest.CONTROL_AE_MODE, aeMode)
                b.set(CaptureRequest.CONTROL_AE_LOCK, aeLock)
                b.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposure.evIndex.coerceIn(info.evRange.lower, info.evRange.upper))
                b.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
            }
            is ExposurePlan.Manual, is ExposurePlan.PerFrame -> {
                val m = if (exposure is ExposurePlan.Manual) exposure.exposure else (exposure as ExposurePlan.PerFrame).at(0)
                val (iso, exp) = if (preview) previewExposure(info, m) else m.iso to m.exposureNs
                b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exp)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, exp)
                // A capture's flash comes resolved (Auto that will not fire arrives as Off).
                val fire = !preview && info.hasFlash && flash != FlashMode.OFF
                b.set(CaptureRequest.FLASH_MODE, if (fire) CameraMetadata.FLASH_MODE_SINGLE else CameraMetadata.FLASH_MODE_OFF)
            }
        }
        if (info.hasOis) {
            b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_ON)
        }
        applyZoom(b, info, zoom)
    }

    private fun applyZoom(b: CaptureRequest.Builder, info: CameraInfo, zoom: Float) {
        val z = zoom.coerceIn(info.zoomRange.lower, info.zoomRange.upper)
        if (info.characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) != null) {
            b.set(CaptureRequest.CONTROL_ZOOM_RATIO, z)
        } else if (z > 1f) {
            val a = info.activeArray
            val w = (a.width() / z).roundToInt()
            val h = (a.height() / z).roundToInt()
            b.set(CaptureRequest.SCALER_CROP_REGION, Rect(a.centerX() - w / 2, a.centerY() - h / 2, a.centerX() + w / 2, a.centerY() + h / 2))
        }
    }

    /** Long manual exposures would stall the viewfinder; preview trades them for gain at the same brightness. */
    private fun previewExposure(info: CameraInfo, m: ManualExposure): Pair<Int, Long> {
        if (m.exposureNs <= MAX_PREVIEW_EXPOSURE_NS) return m.iso to m.exposureNs
        val maxIso = info.isoRange?.upper ?: m.iso
        val iso = (m.iso * m.exposureNs.toDouble() / MAX_PREVIEW_EXPOSURE_NS).roundToInt().coerceAtMost(maxIso)
        return iso to MAX_PREVIEW_EXPOSURE_NS
    }

    /**
     * Settles focus before a burst and holds it for every frame: frames focused differently do
     * not merge. Continuous AF is triggered into a lock (true: the caller cancels it afterwards);
     * a tap has already started its own scan, which is only waited for. Gives up after
     * [AF_LOCK_TIMEOUT_MS] and shoots anyway.
     */
    private suspend fun lockFocus(s: CameraCaptureSession, info: CameraInfo): Boolean {
        if (!info.hasAutoFocus) return false
        val tapped = focusRegion != null
        val done = CompletableDeferred<Unit>()
        afWatcher = { af ->
            if (af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED || af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) done.complete(Unit)
        }
        try {
            if (!tapped) {
                val b = previewBuilder() ?: return false
                b.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
                s.capture(b.build(), null, handler)
            }
            val start = System.nanoTime()
            val locked = withTimeoutOrNull(AF_LOCK_TIMEOUT_MS) { done.await() } != null
            Log.i(TAG, if (locked) "focus locked in ${(System.nanoTime() - start) / 1_000_000}ms" else "focus did not lock in ${AF_LOCK_TIMEOUT_MS}ms")
        } finally {
            afWatcher = null
        }
        return !tapped
    }

    /**
     * Follows the largest face with the 3A regions while no tap holds them. Moves smaller than
     * [FACE_MOVE] are ignored and the region changes at most every [FACE_UPDATE_MS], so the
     * repeating request is not rebuilt every frame; a face gone for [FACE_LOST_MS] lets go.
     */
    private fun followFaces(info: CameraInfo, faces: Array<Face>?) {
        val now = System.currentTimeMillis()
        val face = faces?.filter { it.score >= MIN_FACE_SCORE }?.maxByOrNull { it.bounds.width() * it.bounds.height() }
        val a = info.activeArray
        val next = face?.let {
            val r = it.bounds
            val half = (max(r.width(), r.height()) * FACE_REGION_SCALE / 2).roundToInt()
            val left = (r.centerX() - half).coerceIn(0, a.width() - 1)
            val top = (r.centerY() - half).coerceIn(0, a.height() - 1)
            MeteringRectangle(left, top, min(2 * half, a.width() - left), min(2 * half, a.height() - top), MeteringRectangle.METERING_WEIGHT_MAX - 1)
        }
        if (next != null) faceSeenAt = now
        val current = faceRegion
        val changed = when {
            next == null -> current != null && now - faceSeenAt > FACE_LOST_MS
            current == null -> true
            else -> {
                val moved = hypot((next.x + next.width / 2f - current.x - current.width / 2f).toDouble(), (next.y + next.height / 2f - current.y - current.height / 2f).toDouble())
                moved > a.width() * FACE_MOVE || abs(next.width - current.width) > current.width * FACE_MOVE * 4
            }
        }
        if (!changed || now - faceRegionAt < FACE_UPDATE_MS) return
        faceRegion = next
        faceRegionAt = now
        if (focusRegion != null) return
        scope.launch {
            if (!mutex.tryLock()) return@launch
            try {
                if (_state.value is State.Previewing) runCatching { startPreview() }
            } finally {
                mutex.unlock()
            }
        }
    }

    /** Locks AE once it has converged on the tap region (or after [AE_LOCK_MAX_MS] regardless). */
    private fun maybeLockAe(ae: Int?) {
        if (aeLockDeadline == 0L || aeLock) return
        val now = System.currentTimeMillis()
        val settled = ae == CaptureResult.CONTROL_AE_STATE_CONVERGED || ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED
        if (now < aeLockAfter || (!settled && now < aeLockDeadline)) return
        aeLockDeadline = 0L
        scope.launch {
            mutex.withLock {
                if (focusRegion == null || _state.value !is State.Previewing) return@withLock
                aeLock = true
                runCatching { startPreview() }
                Log.i(TAG, "AE locked at ISO ${_live.value.iso} ${_live.value.exposureNs?.div(1000)}us")
            }
        }
    }

    private fun faceDetectMode(info: CameraInfo): Int {
        val modes = info.characteristics.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES) ?: return CameraMetadata.STATISTICS_FACE_DETECT_MODE_OFF
        return when {
            CameraMetadata.STATISTICS_FACE_DETECT_MODE_FULL in modes -> CameraMetadata.STATISTICS_FACE_DETECT_MODE_FULL
            CameraMetadata.STATISTICS_FACE_DETECT_MODE_SIMPLE in modes -> CameraMetadata.STATISTICS_FACE_DETECT_MODE_SIMPLE
            else -> CameraMetadata.STATISTICS_FACE_DETECT_MODE_OFF
        }
    }

    private suspend fun runPrecapture(s: CameraCaptureSession) {
        val b = previewBuilder() ?: return
        b.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CameraMetadata.CONTROL_AE_PRECAPTURE_TRIGGER_START)
        val done = CompletableDeferred<Unit>()
        var sawPrecapture = false
        aeWatcher = { ae ->
            if (ae == CaptureResult.CONTROL_AE_STATE_PRECAPTURE) sawPrecapture = true
            val settled = ae == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
                ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED ||
                ae == CaptureResult.CONTROL_AE_STATE_LOCKED
            if (sawPrecapture && settled) done.complete(Unit)
        }
        try {
            s.capture(b.build(), null, handler)
            withTimeoutOrNull(PRECAPTURE_TIMEOUT_MS) { done.await() }
        } finally {
            aeWatcher = null
        }
    }

    private val previewCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            val af = result.get(CaptureResult.CONTROL_AF_STATE)
            val ae = result.get(CaptureResult.CONTROL_AE_STATE)
            aeWatcher?.invoke(ae)
            afWatcher?.invoke(af)
            maybeLockAe(ae)
            val info = camera
            val faces = result.get(CaptureResult.STATISTICS_FACES)
            if (info != null) followFaces(info, faces)
            val a = info?.activeArray
            _live.value = LiveExposure(
                focusDioptres = if (info?.hasAutoFocus == true) result.get(CaptureResult.LENS_FOCUS_DISTANCE) else null,
                faces = if (a == null) emptyList() else faces.orEmpty().filter { it.score >= MIN_FACE_SCORE }.map {
                    RectF(
                        it.bounds.left / a.width().toFloat(),
                        it.bounds.top / a.height().toFloat(),
                        it.bounds.right / a.width().toFloat(),
                        it.bounds.bottom / a.height().toFloat(),
                    )
                },
                iso = result.get(CaptureResult.SENSOR_SENSITIVITY),
                exposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                focusLocked = af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED || af == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED,
                focusing = af == CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN || af == CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN,
                aeConverged = ae == CaptureResult.CONTROL_AE_STATE_CONVERGED,
                flashRequired = ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED,
            )
        }
    }

    private fun captureTimeoutMs(spec: CaptureSpec): Long {
        val expMs = when (val e = spec.exposure) {
            is ExposurePlan.Manual -> e.exposure.exposureNs / 1_000_000
            is ExposurePlan.PerFrame -> e.frames.maxOf { it.exposureNs } / 1_000_000
            is ExposurePlan.Auto -> 0L
        }
        return 4_000L + spec.frameCount * max(100L, expMs * 2)
    }

    private fun meteringRect(active: Rect, u: Float, v: Float): MeteringRectangle {
        val half = (min(active.width(), active.height()) * 0.06f).roundToInt()
        val cx = active.left + (u.coerceIn(0f, 1f) * active.width()).roundToInt()
        val cy = active.top + (v.coerceIn(0f, 1f) * active.height()).roundToInt()
        val left = (cx - half).coerceIn(active.left, active.right - 2 * half)
        val top = (cy - half).coerceIn(active.top, active.bottom - 2 * half)
        return MeteringRectangle(left - active.left, top - active.top, 2 * half, 2 * half, MeteringRectangle.METERING_WEIGHT_MAX - 1)
    }

    /**
     * Pairs RAW images with their results by sensor timestamp. Each image is copied out and closed
     * on arrival, so a burst longer than the reader's maxImages never starves the HAL. Copies live
     * in [NativeMemory]: the frames handed to [await]'s caller are theirs, [release] frees the rest.
     */
    private class CaptureCollector(private val expected: Int) {
        private val buffers = HashMap<Long, Triple<ByteBuffer, Int, Int>>()
        private val results = HashMap<Long, TotalCaptureResult>()
        private val done = CompletableDeferred<List<RawFrame>>()
        private val handedOver = HashSet<Long>()
        private var released = false

        val callback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                results[ts] = result
                maybeFinish()
            }

            override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
                fail(CameraFailure(Reason.DEVICE))
            }
        }

        fun onImage(image: Image) {
            image.use {
                if (released || done.isCompleted) return
                buffers[it.timestamp] = Triple(copyRaw(it), it.width, it.height)
            }
            maybeFinish()
        }

        fun fail(e: Throwable) {
            done.completeExceptionally(e)
        }

        suspend fun await(): List<RawFrame> = done.await()

        fun release() {
            released = true
            buffers.forEach { (ts, b) -> if (ts !in handedOver) NativeMemory.free(b.first) }
            buffers.clear()
        }

        private fun maybeFinish() {
            if (done.isCompleted) return
            val matched = buffers.keys.filter { it in results }
            if (matched.size < expected) return
            val frames = matched.sorted().take(expected).map { ts ->
                val (data, w, h) = buffers.getValue(ts)
                handedOver += ts
                RawFrame(data, w, h, ts, results.getValue(ts))
            }
            done.complete(frames)
        }

        private fun copyRaw(image: Image): ByteBuffer {
            val plane = image.planes[0]
            val src = plane.buffer
            val rowBytes = image.width * 2
            val out = NativeMemory.allocate(rowBytes * image.height)
            if (plane.rowStride == rowBytes) {
                src.rewind()
                out.put(src)
            } else {
                for (y in 0 until image.height) {
                    val start = y * plane.rowStride
                    src.limit(start + rowBytes)
                    src.position(start)
                    out.put(src)
                }
            }
            out.rewind()
            return out
        }
    }

    companion object {
        private const val TAG = "PocoPawCamera"
        private const val RAW_MAX_IMAGES = 8
        private const val MAX_RECOVERIES = 3
        private const val SERVICE_RESTART_DELAY_MS = 2_000L
        private const val PRECAPTURE_TIMEOUT_MS = 1_500L
        private const val MAX_PREVIEW_EXPOSURE_NS = 66_666_666L
        private const val AF_LOCK_TIMEOUT_MS = 1_000L
        /** AE gets this long on the tap region before its CONVERGED counts, and at most the second. */
        private const val AE_LOCK_MIN_MS = 300L
        private const val AE_LOCK_MAX_MS = 1_200L
        private const val MIN_FACE_SCORE = 50
        private const val FACE_REGION_SCALE = 1.2f
        /** Of the active array's width. */
        private const val FACE_MOVE = 0.04f
        private const val FACE_UPDATE_MS = 300L
        private const val FACE_LOST_MS = 600L

        fun orientationFor(info: CameraInfo, deviceRotation: Int): Int = when (info.facing) {
            LensFacing.FRONT -> (info.sensorOrientation - deviceRotation + 360) % 360
            else -> (info.sensorOrientation + deviceRotation) % 360
        }

        private fun reasonOfError(error: Int): Reason = when (error) {
            CameraDevice.StateCallback.ERROR_CAMERA_IN_USE -> Reason.IN_USE
            CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE -> Reason.MAX_IN_USE
            CameraDevice.StateCallback.ERROR_CAMERA_DISABLED -> Reason.DISABLED
            CameraDevice.StateCallback.ERROR_CAMERA_DEVICE -> Reason.DEVICE
            CameraDevice.StateCallback.ERROR_CAMERA_SERVICE -> Reason.SERVICE
            else -> Reason.UNKNOWN
        }

        private fun reasonOfAccess(reason: Int): Reason = when (reason) {
            CameraAccessException.CAMERA_IN_USE -> Reason.IN_USE
            CameraAccessException.MAX_CAMERAS_IN_USE -> Reason.MAX_IN_USE
            CameraAccessException.CAMERA_DISABLED -> Reason.DISABLED
            CameraAccessException.CAMERA_DISCONNECTED -> Reason.DISCONNECTED
            CameraAccessException.CAMERA_ERROR -> Reason.DEVICE
            else -> Reason.UNKNOWN
        }
    }
}
