package com.izum1k.pocopaw.camera

import android.annotation.SuppressLint
import android.graphics.ImageFormat
import android.graphics.Rect
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
import java.nio.ByteOrder
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
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
    private var activeCapture: CaptureCollector? = null
    private var aeWatcher: ((Int?) -> Unit)? = null
    private var recoveries = 0

    suspend fun open(info: CameraInfo, texture: SurfaceTexture, controls: PreviewControls) = withContext(dispatcher) {
        mutex.withLock {
            this@CameraController.controls = controls
            focusRegion = null
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

    fun updateControls(next: PreviewControls) {
        scope.launch {
            mutex.withLock {
                controls = next
                if (_state.value is State.Previewing) startPreview()
            }
        }
    }

    /** Focus and meter at a point in normalised sensor coordinates (0..1, sensor's own orientation). */
    fun focusAt(u: Float, v: Float) {
        scope.launch {
            mutex.withLock {
                val info = camera ?: return@withLock
                val s = session ?: return@withLock
                if (_state.value !is State.Previewing) return@withLock
                focusRegion = meteringRect(info.activeArray, u, v)
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

    /** Back to continuous focus and whole-frame metering. */
    fun resetFocus() {
        scope.launch {
            mutex.withLock {
                if (focusRegion == null) return@withLock
                focusRegion = null
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
            try {
                val fires = info.hasFlash && spec.flash != FlashMode.OFF && spec.exposure is ExposurePlan.Auto
                if (fires) runPrecapture(s)
                val requests = List(spec.frameCount) { stillRequest(info, spec, reader.surface) }
                val collector = CaptureCollector(spec.frameCount)
                activeCapture = collector
                generateSequence { reader.acquireNextImage() }.forEach { it.close() }
                reader.setOnImageAvailableListener({ r -> r.acquireNextImage()?.let(collector::onImage) }, handler)
                budget.consume(spec.frameCount)
                s.captureBurst(requests, collector.callback, handler)
                val frames = withTimeout(captureTimeoutMs(spec)) { collector.await() }
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
                activeCapture = null
                reader.setOnImageAvailableListener(null, null)
                if (device != null && session === s) {
                    _state.value = State.Previewing(info.id)
                    runCatching { startPreview() }
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

    private fun stillRequest(info: CameraInfo, spec: CaptureSpec, raw: Surface): CaptureRequest {
        val d = device ?: throw CameraFailure(Reason.UNKNOWN)
        return d.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            previewSurface?.let(::addTarget)
            addTarget(raw)
            applyCommon(this, info, spec.exposure, spec.flash, spec.zoomRatio, preview = false)
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
        val region = focusRegion
        if (info.hasAutoFocus) {
            if (region != null) {
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
                if ((c.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0) > 0) b.set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(region))
            } else {
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            }
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
                b.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposure.evIndex.coerceIn(info.evRange.lower, info.evRange.upper))
                b.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
            }
            is ExposurePlan.Manual -> {
                val (iso, exp) = if (preview) previewExposure(info, exposure.exposure) else exposure.exposure.iso to exposure.exposure.exposureNs
                b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exp)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, exp)
                val fire = !preview && info.hasFlash && flash == FlashMode.ON
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
            _live.value = LiveExposure(
                iso = result.get(CaptureResult.SENSOR_SENSITIVITY),
                exposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                focusLocked = af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED || af == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED,
                focusing = af == CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN || af == CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN,
                aeConverged = ae == CaptureResult.CONTROL_AE_STATE_CONVERGED,
            )
        }
    }

    private fun captureTimeoutMs(spec: CaptureSpec): Long {
        val expMs = (spec.exposure as? ExposurePlan.Manual)?.exposure?.exposureNs?.div(1_000_000) ?: 0L
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
     * on arrival, so a burst longer than the reader's maxImages never starves the HAL.
     */
    private class CaptureCollector(private val expected: Int) {
        private val buffers = HashMap<Long, Triple<ByteBuffer, Int, Int>>()
        private val results = HashMap<Long, TotalCaptureResult>()
        private val done = CompletableDeferred<List<RawFrame>>()

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
            image.use { buffers[it.timestamp] = Triple(copyRaw(it), it.width, it.height) }
            maybeFinish()
        }

        fun fail(e: Throwable) {
            done.completeExceptionally(e)
        }

        suspend fun await(): List<RawFrame> = done.await()

        private fun maybeFinish() {
            val matched = buffers.keys.filter { it in results }
            if (matched.size < expected) return
            val frames = matched.sorted().take(expected).map { ts ->
                val (data, w, h) = buffers.getValue(ts)
                RawFrame(data, w, h, ts, results.getValue(ts))
            }
            done.complete(frames)
        }

        private fun copyRaw(image: Image): ByteBuffer {
            val plane = image.planes[0]
            val src = plane.buffer
            val rowBytes = image.width * 2
            val out = ByteBuffer.allocateDirect(rowBytes * image.height).order(ByteOrder.nativeOrder())
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
