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
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.Face
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.RggbChannelVector
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.pipeline.NativeMemory
import com.izum1k.pocopaw.pipeline.RawBurst
import com.izum1k.pocopaw.pipeline.RawFrame
import com.izum1k.pocopaw.pipeline.RawMetadata
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.android.asCoroutineDispatcher
import kotlinx.coroutines.channels.Channel
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
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class CameraFailure(val reason: CameraController.Reason, cause: Throwable? = null) : Exception(reason.name, cause)

/**
 * The only owner of a CameraDevice. Preview requests carry the YUV preview surface only; capture
 * requests add the RAW surface and are counted against [budget]. Every Camera2 call runs on one
 * handler thread, serialised by a mutex.
 */
class CameraController(private val manager: CameraManager, private val motion: MotionMeter? = null) {

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
    /** AWB's latest gains and matrix, and the pair a white balance shift is applied to (frozen when it starts). */
    private var autoWb: Pair<RggbChannelVector, ColorSpaceTransform>? = null
    private var wbBase: Pair<RggbChannelVector, ColorSpaceTransform>? = null
    private var wbCameraId: String? = null
    private var activeCapture: CaptureCollector? = null
    private var aeWatcher: ((Int?) -> Unit)? = null
    private var afWatcher: ((CaptureResult) -> Unit)? = null
    /** The tap's AF scan being checked; null once it has locked (or been given up on). */
    private var tapScan: TapScan? = null
    private var recoveries = 0
    private var lastAfState: Int? = null
    /** Small YUV stream the app measures focus on; only sweep requests carry it. Null if the session could not take it. */
    private var yuvReader: ImageReader? = null
    private var sweep: Sweep? = null
    /** Lens distance a tap's sweep settled on, held with AF off until the tap is released. */
    private var tapFocus: Float? = null
    /** Lens distance the scene's sweep settled on, held with AF off while no tap overrides it. */
    private var sceneFocus: Float? = null
    /** ISO x exposure when [sceneFocus] was found, to tell how much the scene changed since. */
    private var sceneExposure: Double? = null
    private var sceneSweepAfter = 0L
    private var sceneSweepDeadline = 0L
    /** A tap waits for AE on its region before AF scans: when it was made, 0 when none waits. */
    private var tapPendingSince = 0L
    private var moving = false
    private var movedSinceNs = 0L

    private class TapScan(val fromDioptres: Float?, val retry: Boolean) {
        var triggered = false
        var scanned = false
    }

    /** A running focus sweep: [target] is the lens distance its current [step] asks for. */
    private class Sweep(val zones: List<FocusZone>, var target: Float) {
        var step = -1
        var abort = false
        var job: Job? = null
        var scoredNs = 0L
        var scored = 0
        val samples = Channel<FocusSample>(Channel.UNLIMITED)
        val images = HashMap<Long, FloatArray>()
        val results = HashMap<Long, Pair<SweepTag, CaptureResult>>()
    }

    private class SweepTag(val sweep: Sweep, val step: Int, val target: Float)

    suspend fun open(info: CameraInfo, texture: SurfaceTexture, controls: PreviewControls) = withContext(dispatcher) {
        mutex.withLock {
            this@CameraController.controls = controls
            focusRegion = null
            tapScan = null
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
            openLocked(info, tex, keepFocus = true)
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
     * holds it, meters there and locks AE once it has settled, until [resetFocus]. AF scans once AE
     * has settled on the point (a scan while the exposure ramps finds false peaks); with the YUV
     * stream, the app then sweeps around where AF stopped and holds the sharpest distance.
     */
    fun focusAt(u: Float, v: Float) {
        scope.launch {
            mutex.withLock {
                val info = camera ?: return@withLock
                if (session == null || _state.value !is State.Previewing) return@withLock
                cancelSweep()
                tapFocus = null
                tapScan = null
                sceneSweepAfter = 0L
                focusRegion = meteringRect(info.activeArray, u, v)
                aeLock = false
                val now = System.currentTimeMillis()
                aeLockAfter = now + AE_LOCK_MIN_MS
                aeLockDeadline = now + AE_LOCK_MAX_MS
                tapPendingSince = now
                runCatching { startPreview() }
            }
        }
    }

    /** Starts a waiting tap's AF scan once AE has settled on its region, or after [AE_SETTLE_MAX_MS]. */
    private fun maybeStartTapScan(ae: Int?) {
        if (tapPendingSince == 0L || _state.value !is State.Previewing) return
        val waited = System.currentTimeMillis() - tapPendingSince
        if (waited < AE_SETTLE_MIN_MS || (!aeSettled(ae) && waited < AE_SETTLE_MAX_MS)) return
        tapPendingSince = 0L
        val s = session ?: return
        startTapScan(s, TapScan(_live.value.focusDioptres, retry = false))
    }

    private fun aeSettled(ae: Int?) = ae == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
        ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED ||
        ae == CaptureResult.CONTROL_AE_STATE_LOCKED

    private fun startTapScan(s: CameraCaptureSession, t: TapScan) {
        tapScan = t
        runCatching {
            retriggerFocus(s) { r ->
                if (tapScan !== t) return@retriggerFocus
                t.triggered = true
                checkTapScan(r)
            }
        }.onFailure { Log.w(TAG, "focus failed", it) }
    }

    /** Cancels whatever AF holds and starts a scan on the current regions; [onTrigger] gets the trigger's result. */
    private fun retriggerFocus(s: CameraCaptureSession, onTrigger: (CaptureResult) -> Unit) {
        val cancel = previewBuilder() ?: return
        cancel.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)
        s.capture(cancel.build(), null, handler)
        startPreview()
        val trigger = previewBuilder() ?: return
        trigger.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
        s.capture(trigger.build(), triggerCallback(onTrigger), handler)
    }

    private fun triggerCallback(onResult: (CaptureResult) -> Unit) = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) = onResult(result)
    }

    /**
     * A tap's scan can end without focusing, or lock at once where the lens already was (no scan
     * seen, distance unchanged), which leaves the shot soft. Either way it is triggered once more.
     * Results before the trigger's own still show the previous state and are skipped.
     */
    private fun checkTapScan(result: CaptureResult) {
        val t = tapScan ?: return
        if (!t.triggered) return
        val af = result.get(CaptureResult.CONTROL_AF_STATE)
        if (af == CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN) t.scanned = true
        if (af != CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED && af != CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) return
        val to = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
        val from = t.fromDioptres
        val unmoved = from != null && to != null && abs(to - from) < FOCUS_UNMOVED_DIOPTRES
        val failed = af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED || (unmoved && !t.scanned)
        val retry = failed && !t.retry
        Log.i(
            TAG,
            "tap focus ${if (af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED) "locked" else "not focused"} " +
                "%.2f -> %.2f D, scanned=${t.scanned}${if (retry) ", retriggering" else ""}".format(from ?: Float.NaN, to ?: Float.NaN),
        )
        val next = if (retry) TapScan(to, retry = true) else null
        tapScan = next
        if (next == null) {
            if (_state.value is State.Previewing) startTapSweep(to, focused = af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED)
            return
        }
        scope.launch {
            mutex.withLock {
                val s = session ?: return@withLock
                // a capture, reset or new tap since has taken focus over
                if (tapScan !== next || focusRegion == null || _state.value !is State.Previewing) return@withLock
                startTapScan(s, next)
            }
        }
    }

    /** Back to the scene's held distance (or continuous focus), whole-frame metering and running AE. */
    fun resetFocus() {
        scope.launch {
            mutex.withLock {
                if (focusRegion == null && !aeLock && tapFocus == null) return@withLock
                if (focusRegion != null) cancelSweep()
                focusRegion = null
                tapScan = null
                tapFocus = null
                tapPendingSince = 0L
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

    /** The distance held with AF off: a swept tap's, else the scene's while no tap holds the regions. */
    private fun heldFocus(): Float? = tapFocus ?: if (focusRegion == null) sceneFocus else null

    /** After a tap's AF has stopped at [d0]: a fine sweep there, or a full one when AF failed or the fine one finds no clear peak. */
    private fun startTapSweep(d0: Float?, focused: Boolean) {
        val info = camera ?: return
        val region = focusRegion ?: return
        if (yuvReader == null) return
        val a = info.activeArray
        val zone = FocusZone(
            RectF(region.x / a.width().toFloat(), region.y / a.height().toFloat(), (region.x + region.width) / a.width().toFloat(), (region.y + region.height) / a.height().toFloat()),
            1f,
        )
        val sw = Sweep(listOf(zone), d0 ?: 0f)
        launchSweep(
            sw,
            run = { (if (focused && d0 != null) fineSweep(sw, info, d0) else null) ?: fullSweep(sw, info) },
            done = { r -> if (focusRegion === region) tapFocus = r?.dioptres },
        )
    }

    /** Sweeps the scene (the largest face, else the middle of the frame): fully, or finely around [d0]. */
    private fun startSceneSweep(d0: Float?) {
        val info = camera ?: return
        if (yuvReader == null) return
        val a = info.activeArray
        val zones = faceRegion?.let { f ->
            listOf(FocusZone(RectF(f.x / a.width().toFloat(), f.y / a.height().toFloat(), (f.x + f.width) / a.width().toFloat(), (f.y + f.height) / a.height().toFloat()), 1f))
        } ?: FocusSweeper.centreZones()
        val sw = Sweep(zones, d0 ?: 0f)
        launchSweep(
            sw,
            run = { (if (d0 != null) fineSweep(sw, info, d0) else null) ?: fullSweep(sw, info) },
            done = { r ->
                sceneFocus = r?.dioptres
                sceneExposure = exposureProduct()
            },
        )
    }

    /** The scene's first sweep, once AE has settled after opening (or after [SCENE_SWEEP_MAX_WAIT_MS]). */
    private fun maybeSweepScene(ae: Int?) {
        if (sceneSweepAfter == 0L || sweep != null || focusRegion != null || moving || _state.value !is State.Previewing) return
        val now = System.currentTimeMillis()
        if (now < sceneSweepAfter || (!aeSettled(ae) && now < sceneSweepDeadline)) return
        sceneSweepAfter = 0L
        startSceneSweep(null)
    }

    /**
     * Turning the phone gives focus back to continuous AF so the viewfinder follows; once it has
     * been still for [SETTLE_WINDOW_NS], a small turn at about the same exposure is swept finely
     * around where AF went, anything bigger fully. A tap's hold is left alone.
     */
    private fun followMotion() {
        val m = motion ?: return
        if (focusRegion != null || yuvReader == null || _state.value !is State.Previewing) return
        val fast = m.rms(MOVE_WINDOW_NS) ?: return
        if (fast > MOVING_RAD_PER_SEC) {
            if (moving) return
            moving = true
            movedSinceNs = m.newestNs() ?: 0L
            sceneSweepAfter = 0L
            val held = sceneFocus != null || sweep != null
            cancelSweep()
            sceneFocus = null
            Log.i(TAG, "moving (%.3f rad/s): continuous AF".format(fast))
            if (held) runCatching { startPreview() }
            return
        }
        if (!moving) return
        val calm = m.rms(SETTLE_WINDOW_NS) ?: return
        if (calm > STILL_RAD_PER_SEC) return
        moving = false
        val angle = m.angleSince(movedSinceNs)
        val before = sceneExposure
        val now = exposureProduct()
        val ev = if (before != null && now != null) ln(now / before) / ln(2.0) else null
        val small = angle < SMALL_TURN_RAD && ev != null && abs(ev) < SMALL_EV_CHANGE
        Log.i(TAG, "settled after %.2f rad, %.2f EV: ${if (small) "fine" else "full"} sweep".format(angle, ev ?: Double.NaN))
        startSceneSweep(if (small) _live.value.focusDioptres else null)
    }

    private fun exposureProduct(): Double? {
        val l = _live.value
        return (l.iso ?: return null).toDouble() * (l.exposureNs ?: return null)
    }

    /**
     * Runs [sw] in its own coroutine. Each step takes the mutex only to rebuild the repeating
     * request; [done] gets the result under the mutex unless the sweep was cancelled meanwhile.
     */
    private fun launchSweep(sw: Sweep, run: suspend () -> SweepResult?, done: (SweepResult?) -> Unit) {
        cancelSweep()
        sweep = sw
        sw.job = scope.launch {
            val result = try {
                run()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.w(TAG, "focus sweep failed", e)
                null
            }
            mutex.withLock {
                if (sweep !== sw) return@withLock
                sweep = null
                done(result)
                if (result == null) Log.i(TAG, "sweep found no peak, AF keeps focus")
                if (_state.value is State.Previewing) runCatching { startPreview() }
            }
        }
    }

    /** Drops the running sweep, if any (true); the caller rebuilds the repeating request. */
    private fun cancelSweep(): Boolean {
        val sw = sweep ?: return false
        sweep = null
        sw.job?.cancel()
        return true
    }

    /** A press during a sweep waits up to [REFINE_MAX_MS], then makes it stop at the best step so far. */
    private suspend fun awaitSweep() {
        val job = sweep?.job ?: return
        if (withTimeoutOrNull(REFINE_MAX_MS) { job.join() } != null) return
        sweep?.abort = true
        Log.i(TAG, "capture cuts the sweep short")
        withTimeoutOrNull(STEP_TIMEOUT_MS * 2) { job.join() }
    }

    /**
     * Asks for each distance in [steps] with AF off, [frames] frames each, in one burst: a request
     * at a time would wait out the HAL's pipeline at every step. Keeps each step's last frame, the
     * lens most settled. A deadline or abort flushes the rest of the burst.
     */
    private suspend fun sweepSteps(sw: Sweep, steps: List<Float>, frames: Int, deadline: Long): List<FocusSample> {
        if (steps.isEmpty()) return emptyList()
        val first = sw.step + 1
        val lastStep = first + steps.lastIndex
        val total = mutex.withLock {
            val s = session
            val yuv = yuvReader
            if (sweep !== sw || s == null || yuv == null || _state.value !is State.Previewing) throw CancellationException("sweep superseded")
            val requests = steps.flatMapIndexed { k, d ->
                val n = if (k == 0) frames + FIRST_STEP_EXTRA_FRAMES else frames
                val b = previewBuilder() ?: throw CancellationException("no device")
                b.addTarget(yuv.surface)
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                b.set(CaptureRequest.LENS_FOCUS_DISTANCE, d)
                b.setTag(SweepTag(sw, first + k, d))
                val r = b.build()
                List(n) { r }
            }
            sw.step = lastStep
            sw.target = steps.last()
            s.captureBurst(requests, previewCallback, handler)
            startPreview()
            requests.size
        }
        val last = HashMap<Int, FocusSample>()
        var received = 0
        while (received < total) {
            val left = deadline - System.currentTimeMillis()
            if (sw.abort || left <= 0) {
                mutex.withLock {
                    if (sweep === sw) runCatching {
                        session?.abortCaptures()
                        startPreview()
                    }
                }
                break
            }
            val s = withTimeoutOrNull(min(left, STEP_TIMEOUT_MS)) { sw.samples.receive() } ?: break
            if (s.step < first || s.step > lastStep) continue
            received++
            last[s.step] = s
        }
        return (first..lastStep).mapNotNull { last[it] }
    }

    /** Far to near in coarse steps, then fine steps between the coarse peak's neighbours. */
    private suspend fun fullSweep(sw: Sweep, info: CameraInfo): SweepResult? {
        if (sw.abort) return null
        val start = System.currentTimeMillis()
        val minFocus = minFocusDioptres(info)
        val coarse = sweepSteps(sw, FocusSweeper.coarseSteps(minFocus), COARSE_FRAMES, start + FULL_SWEEP_MAX_MS)
        logCurve("coarse", sw, coarse)
        if (coarse.size < 3) return null
        val contrast = FocusSweeper.contrast(coarse, sw.zones)
        val i = FocusSweeper.pick(coarse, sw.zones)
        if (contrast < FocusSweeper.FLAT_CONTRAST) {
            logSweep("full", null, null, contrast, coarse.size, start)
            return null
        }
        if (sw.abort) return logSweep("full", null, coarse[i].dioptres, contrast, coarse.size, start)
        val h = fineStep(info)
        val lo = coarse.getOrNull(i - 1)?.target ?: (coarse[i].target - h)
        val hi = coarse.getOrNull(i + 1)?.target ?: (coarse[i].target + h)
        val fine = sweepSteps(sw, FocusSweeper.fineSteps(lo, hi, h, minFocus), FINE_FRAMES, System.currentTimeMillis() + REFINE_MAX_MS)
        logCurve("fine", sw, fine)
        if (fine.size < 3) return logSweep("full", null, coarse[i].dioptres, contrast, coarse.size + fine.size, start)
        val d = FocusSweeper.fitPeak(fine, sw.zones) ?: FocusSweeper.interpolate(fine, sw.zones, FocusSweeper.pick(fine, sw.zones))
        return logSweep("full", null, d, contrast, coarse.size + fine.size, start)
    }

    /**
     * Fine steps around [d0]; a peak on an edge short of the lens's range is followed outwards a
     * few steps at a time. Null when the curve is flat or the peak runs away, for a full sweep.
     */
    private suspend fun fineSweep(sw: Sweep, info: CameraInfo, d0: Float): SweepResult? {
        val start = System.currentTimeMillis()
        val minFocus = minFocusDioptres(info)
        val h = fineStep(info)
        val samples = sweepSteps(sw, FocusSweeper.fineSteps(d0 - FINE_HALF_STEPS * h, d0 + FINE_HALF_STEPS * h, h, minFocus), FINE_FRAMES, start + REFINE_MAX_MS)
            .toMutableList()
        var extensions = 0
        while (true) {
            if (samples.size < 3) return null
            samples.sortBy { it.target }
            val i = FocusSweeper.pick(samples, sw.zones)
            val lowEdge = i == 0 && samples.first().target > 0.01f
            val highEdge = i == samples.lastIndex && samples.last().target < minFocus - 0.01f
            if (sw.abort || (!lowEdge && !highEdge)) break
            if (extensions == MAX_FINE_EXTENSIONS) {
                logCurve("fine", sw, samples)
                logSweep("fine", d0, null, FocusSweeper.contrast(samples, sw.zones), samples.size, start)
                return null
            }
            extensions++
            val from = if (lowEdge) samples.first().target else samples.last().target
            val dir = if (lowEdge) -1 else 1
            val more = (1..FINE_EXTENSION_STEPS).map { (from + dir * it * h).coerceIn(0f, minFocus) }.distinct()
            samples += sweepSteps(sw, more, FINE_FRAMES, System.currentTimeMillis() + REFINE_MAX_MS)
        }
        logCurve("fine", sw, samples)
        // a fine sweep spans a fraction of the peak, so it is judged by its shape, not by max / min
        val contrast = FocusSweeper.contrast(samples, sw.zones)
        val i = FocusSweeper.pick(samples, sw.zones)
        val atBound = samples[i].target <= 0.01f || samples[i].target >= minFocus - 0.01f
        val d = when {
            contrast < FocusSweeper.FINE_FLAT_CONTRAST -> null
            atBound -> samples[i].dioptres
            else -> FocusSweeper.fitPeak(samples, sw.zones)
                ?: if (i > 0 && i < samples.lastIndex) FocusSweeper.interpolate(samples, sw.zones, i)
                else if (sw.abort) samples[i].dioptres else null
        }
        return logSweep("fine", d0, d, contrast, samples.size, start)
    }

    private fun logSweep(kind: String, d0: Float?, d: Float?, contrast: Float, steps: Int, start: Long): SweepResult? {
        val ms = System.currentTimeMillis() - start
        Log.i(TAG, "sweep $kind %.2f -> %.2f D, peak/floor %.2f, $steps steps, $ms ms".format(d0 ?: Float.NaN, d ?: Float.NaN, contrast))
        return d?.let { SweepResult(it, contrast, steps, ms) }
    }

    private fun logCurve(stage: String, sw: Sweep, samples: List<FocusSample>) {
        if (samples.isEmpty()) return
        val c = FocusSweeper.combined(samples, sw.zones)
        Log.i(
            TAG,
            "sweep $stage curve " + samples.indices.joinToString(" ") { k ->
                val s = samples[k]
                "%.2f/%.2f%s:%.2f".format(s.target, s.dioptres, if (s.stationary) "" else "m", c[k])
            } + ", scoring %.1f ms/frame over ${sw.scored}".format(if (sw.scored == 0) 0.0 else sw.scoredNs / 1e6 / sw.scored),
        )
    }

    private fun minFocusDioptres(info: CameraInfo): Float =
        info.characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE)?.takeIf { it > 0f } ?: 10f

    /** About half the depth of field in dioptres: the hyperfocal distance's dioptres, the same at every distance. */
    private fun fineStep(info: CameraInfo): Float =
        (info.characteristics.get(CameraCharacteristics.LENS_INFO_HYPERFOCAL_DISTANCE) ?: 0.1f).coerceIn(0.05f, 0.3f)

    private fun onYuv(image: Image) {
        val sw = sweep ?: return
        val ts = image.timestamp
        val t0 = System.nanoTime()
        sw.images[ts] = FocusSweeper.score(image, sw.zones)
        sw.scoredNs += System.nanoTime() - t0
        sw.scored++
        matchSweep(sw, ts)
    }

    private fun onSweepResult(tag: SweepTag, result: CaptureResult) {
        val sw = tag.sweep
        if (sweep !== sw) return
        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        sw.results[ts] = tag to result
        matchSweep(sw, ts)
    }

    /** A sample once both the frame's image and its result have arrived (in either order). */
    private fun matchSweep(sw: Sweep, ts: Long) {
        val scores = sw.images[ts]
        val tagged = sw.results[ts]
        if (scores != null && tagged != null) {
            sw.images.remove(ts)
            sw.results.remove(ts)
            val (tag, r) = tagged
            val stationary = r.get(CaptureResult.LENS_STATE) != CaptureResult.LENS_STATE_MOVING
            sw.samples.trySend(FocusSample(tag.step, tag.target, r.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: tag.target, stationary, scores))
        }
        while (sw.images.size > SWEEP_PENDING_MAX) sw.images.remove(sw.images.keys.min())
        while (sw.results.size > SWEEP_PENDING_MAX) sw.results.remove(sw.results.keys.min())
    }

    /** Captures [CaptureSpec.frameCount] RAW frames, reopening the device first if the budget cannot cover them. */
    suspend fun capture(spec: CaptureSpec): RawBurst = withContext(dispatcher) {
        awaitSweep()
        mutex.withLock {
            var info = camera ?: throw CameraFailure(Reason.UNKNOWN)
            if (!budget.canAfford(spec.frameCount)) {
                Log.i(TAG, "RAW budget at ${budget.remaining.value}, reopening ${info.id}")
                openLocked(info, texture ?: throw CameraFailure(Reason.UNKNOWN), keepFocus = true)
                info = camera ?: throw CameraFailure(Reason.UNKNOWN)
            }
            val s = session ?: throw CameraFailure(Reason.UNKNOWN)
            val reader = rawReader ?: throw CameraFailure(Reason.UNKNOWN)
            if (cancelSweep()) runCatching { startPreview() }
            _state.value = State.Capturing(info.id)
            if (tapPendingSince != 0L) {
                tapPendingSince = 0L
                startTapScan(s, TapScan(_live.value.focusDioptres, retry = false))
            }
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
                Log.i(
                    TAG,
                    "burst lens " + frames.joinToString(" ") { f ->
                        val moving = f.result.get(CaptureResult.LENS_STATE) == CaptureResult.LENS_STATE_MOVING
                        "%.2f%s".format(f.result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: Float.NaN, if (moving) "m" else "")
                    } + " (AF ${afStateName(frames.first().result.get(CaptureResult.CONTROL_AF_STATE))})",
                )
                RawBurst(
                    frames = steadyFrames(frames, spec),
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

    /**
     * Opens [info] and starts the preview. A fresh open sweeps the scene once AE has settled;
     * [keepFocus] (a reopen for the RAW budget, same camera, phone still) keeps the scene's distance.
     */
    private suspend fun openLocked(info: CameraInfo, tex: SurfaceTexture, keepFocus: Boolean = false) {
        val kept = if (keepFocus && camera?.id == info.id && !moving) sceneFocus to sceneExposure else null
        closeLocked()
        if (info.id != wbCameraId) {
            autoWb = null
            wbBase = null
            wbCameraId = info.id
        }
        camera = info
        texture = tex
        tapFocus = null
        tapPendingSince = 0L
        sceneFocus = kept?.first
        sceneExposure = kept?.second
        moving = false
        sceneSweepAfter = 0L
        _state.value = State.Opening(info.id)
        try {
            tex.setDefaultBufferSize(info.previewSize.width, info.previewSize.height)
            val surface = Surface(tex).also { previewSurface = it }
            val reader = ImageReader.newInstance(info.rawSize.width, info.rawSize.height, ImageFormat.RAW_SENSOR, RAW_MAX_IMAGES)
            rawReader = reader
            val dev = openDevice(info.id)
            device = dev
            budget.reset()
            session = createSessionWithYuv(dev, info, surface, reader.surface)
            startPreview()
            _state.value = State.Previewing(info.id)
            if (yuvReader != null) {
                motion?.start(this)
                if (sceneFocus == null) {
                    val now = System.currentTimeMillis()
                    sceneSweepAfter = now + SCENE_SWEEP_MIN_MS
                    sceneSweepDeadline = now + SCENE_SWEEP_MAX_WAIT_MS
                }
            }
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
        cancelSweep()
        motion?.stop(this)
        runCatching { session?.close() }
        runCatching { device?.close() }
        runCatching { rawReader?.close() }
        runCatching { yuvReader?.close() }
        runCatching { previewSurface?.release() }
        session = null
        device = null
        rawReader = null
        yuvReader = null
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
     * Preview, RAW and, for a lens that focuses, the YUV analysis stream. A session that will not
     * take the third stream is made without it, and focus is left to the HAL.
     */
    private suspend fun createSessionWithYuv(dev: CameraDevice, info: CameraInfo, preview: Surface, raw: Surface): CameraCaptureSession {
        val size = if (info.hasAutoFocus) analysisSize(info) else null
        if (size == null) return createSession(dev, listOf(preview, raw))
        val yuv = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, YUV_MAX_IMAGES)
        val surfaces = listOf(preview, raw, yuv.surface)
        val supported = runCatching {
            @Suppress("DEPRECATION")
            dev.isSessionConfigurationSupported(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, surfaces.map { OutputConfiguration(it) }, executor, NoopSessionCallback))
        }.getOrDefault(true)
        if (supported) {
            try {
                val s = createSession(dev, surfaces)
                yuv.setOnImageAvailableListener({ r -> r.acquireNextImage()?.use(::onYuv) }, handler)
                yuvReader = yuv
                Log.i(TAG, "focus analysis stream ${size.width}x${size.height}")
                return s
            } catch (e: CameraFailure) {
                if (e.reason != Reason.CONFIGURATION) {
                    yuv.close()
                    throw e
                }
            }
        }
        yuv.close()
        Log.w(TAG, "no focus analysis stream at ${size.width}x${size.height}: focus stays with the HAL")
        return createSession(dev, listOf(preview, raw))
    }

    /** The largest 4:3 YUV size up to [MAX_ANALYSIS_WIDTH]: detail fine enough to judge focus on, cheap to score a part of. */
    private fun analysisSize(info: CameraInfo): Size? {
        val map = info.characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val aspect = info.activeArray.width().toFloat() / info.activeArray.height()
        return map.getOutputSizes(ImageFormat.YUV_420_888)
            ?.filter { abs(it.width.toFloat() / it.height - aspect) < 0.02f && it.width <= MAX_ANALYSIS_WIDTH }
            ?.maxByOrNull { it.width.toLong() * it.height }
    }

    private object NoopSessionCallback : CameraCaptureSession.StateCallback() {
        override fun onConfigured(session: CameraCaptureSession) = Unit
        override fun onConfigureFailed(session: CameraCaptureSession) = Unit
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
            val hold = sweep?.target ?: heldFocus()
            if (hold != null) {
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                b.set(CaptureRequest.LENS_FOCUS_DISTANCE, hold)
            } else {
                b.set(CaptureRequest.CONTROL_AF_MODE, if (tap != null) CameraMetadata.CONTROL_AF_MODE_AUTO else CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            }
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
                b.set(CaptureRequest.CONTROL_AE_LOCK, aeLock || sweep != null)
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
        applyWhiteBalance(b)
        applyZoom(b, info, zoom)
    }

    /**
     * A shift holds AWB's gains from when it started and tilts them warm or cool by [WB_STEP] per
     * step (red up, blue down); stills carry the same gains, so the pipeline follows. Until AWB has
     * reported, and at 0, AWB runs.
     */
    private fun applyWhiteBalance(b: CaptureRequest.Builder) {
        val shift = controls.wbShift
        if (shift == 0) {
            wbBase = null
            b.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
            return
        }
        val base = wbBase ?: autoWb?.also { wbBase = it } ?: run {
            b.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
            return
        }
        val f = Math.pow(WB_STEP, shift.toDouble()).toFloat()
        val g = base.first
        b.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_OFF)
        b.set(CaptureRequest.COLOR_CORRECTION_MODE, CameraMetadata.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX)
        b.set(CaptureRequest.COLOR_CORRECTION_GAINS, RggbChannelVector(g.red * f, g.greenEven, g.greenOdd, g.blue / f))
        b.set(CaptureRequest.COLOR_CORRECTION_TRANSFORM, base.second)
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
     * a tap has already started its own scan, which is waited for and must end focused: a tap
     * left unfocused is scanned once more. Gives up after [AF_LOCK_TIMEOUT_MS] and shoots anyway.
     */
    private suspend fun lockFocus(s: CameraCaptureSession, info: CameraInfo): Boolean {
        if (!info.hasAutoFocus) return false
        heldFocus()?.let { d ->
            tapScan = null
            // a sweep that just ended leaves the lens on its way back to d
            val arrived = CompletableDeferred<Unit>()
            afWatcher = { r ->
                val at = r.get(CaptureResult.LENS_FOCUS_DISTANCE)
                if (r.get(CaptureResult.LENS_STATE) != CaptureResult.LENS_STATE_MOVING && at != null && abs(at - d) <= BURST_FOCUS_TOLERANCE) arrived.complete(Unit)
            }
            val start = System.nanoTime()
            val ok = try {
                withTimeoutOrNull(HOLD_SETTLE_MAX_MS) { arrived.await() } != null
            } finally {
                afWatcher = null
            }
            val ms = (System.nanoTime() - start) / 1_000_000
            Log.i(TAG, "focus held at %.2f D (${if (tapFocus != null) "tap" else "scene"} sweep), lens ${if (ok) "there in ${ms}ms" else "not there"}".format(d))
            return false
        }
        val tapped = focusRegion != null
        tapScan = null
        var done = CompletableDeferred<Int>()
        // after a trigger, results still in flight report the old state until the trigger's own
        var waitTrigger = false
        val onResult = { r: CaptureResult ->
            val af = r.get(CaptureResult.CONTROL_AF_STATE)
            if (!waitTrigger && (af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED || af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED)) done.complete(af)
        }
        val onTrigger = { r: CaptureResult ->
            waitTrigger = false
            onResult(r)
        }
        afWatcher = onResult
        try {
            if (!tapped) {
                val b = previewBuilder() ?: return false
                b.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
                waitTrigger = true
                s.capture(b.build(), triggerCallback(onTrigger), handler)
            }
            val start = System.nanoTime()
            var af = withTimeoutOrNull(AF_LOCK_TIMEOUT_MS) { done.await() }
            if (tapped && af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) {
                Log.i(TAG, "tap focus not focused at capture (%.2f D), retriggering".format(_live.value.focusDioptres ?: Float.NaN))
                done = CompletableDeferred()
                waitTrigger = true
                runCatching { retriggerFocus(s, onTrigger) }.onFailure { Log.w(TAG, "focus failed", it) }
                af = withTimeoutOrNull(AF_LOCK_TIMEOUT_MS) { done.await() }
            }
            val ms = (System.nanoTime() - start) / 1_000_000
            Log.i(
                TAG,
                when (af) {
                    null -> "focus did not lock in ${AF_LOCK_TIMEOUT_MS}ms"
                    CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> "focus locked in ${ms}ms"
                    else -> "focus locked unfocused in ${ms}ms"
                } + " at %.2f D".format(_live.value.focusDioptres ?: Float.NaN),
            )
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

    /** Keeps AWB's output; a shift waiting for it rebuilds the preview once it arrives. */
    private fun trackAutoWb(result: CaptureResult) {
        if (result.get(CaptureResult.CONTROL_AWB_MODE) == CameraMetadata.CONTROL_AWB_MODE_OFF) return
        val gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS) ?: return
        val transform = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM) ?: return
        val first = autoWb == null
        autoWb = gains to transform
        if (first && controls.wbShift != 0 && wbBase == null) {
            scope.launch {
                if (!mutex.tryLock()) return@launch
                try {
                    if (_state.value is State.Previewing) runCatching { startPreview() }
                } finally {
                    mutex.unlock()
                }
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
            if (af != lastAfState) {
                lastAfState = af
                Log.i(TAG, "af ${afStateName(af)} at %.2f D".format(result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: Float.NaN))
            }
            (request.tag as? SweepTag)?.let { onSweepResult(it, result) }
            aeWatcher?.invoke(ae)
            afWatcher?.invoke(result)
            checkTapScan(result)
            maybeLockAe(ae)
            maybeStartTapScan(ae)
            maybeSweepScene(ae)
            followMotion()
            trackAutoWb(result)
            val info = camera
            val faces = result.get(CaptureResult.STATISTICS_FACES)
            if (info != null) followFaces(info, faces)
            val a = info?.activeArray
            val settling = sweep != null || tapPendingSince != 0L
            val held = heldFocus() != null
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
                focusLocked = !settling && (held || af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED || af == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED),
                focusing = settling || af == CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN || af == CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN,
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

    /**
     * Frames shot while the lens moved, or away from where the rest of the burst sat, would merge
     * soft: they are left out, never the first (the metadata's) and never below two frames.
     * Bracketed bursts keep every frame, their exposures go in a fixed pattern.
     */
    private fun steadyFrames(frames: List<RawFrame>, spec: CaptureSpec): List<RawFrame> {
        if (frames.size < 3 || spec.exposure is ExposurePlan.PerFrame) return frames
        val ds = frames.map { it.result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: return frames }
        val median = ds.sorted()[ds.size / 2]
        val off = frames.indices.filter { i ->
            i != 0 && (frames[i].result.get(CaptureResult.LENS_STATE) == CaptureResult.LENS_STATE_MOVING || abs(ds[i] - median) > BURST_FOCUS_TOLERANCE)
        }
        if (off.isEmpty() || frames.size - off.size < 2) return frames
        Log.w(TAG, "leaving out ${off.size} of ${frames.size} frames shot off focus (median %.2f D)".format(median))
        off.forEach { frames[it].release() }
        return frames.filterIndexed { i, _ -> i !in off }
    }

    private fun meteringRect(active: Rect, u: Float, v: Float): MeteringRectangle {
        val half = (min(active.width(), active.height()) * TAP_REGION_HALF).roundToInt()
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
        /** A tap's lock this close to where the lens was, with no scan seen, never moved the lens. */
        private const val FOCUS_UNMOVED_DIOPTRES = 0.02f
        /** AE gets this long on the tap region before its CONVERGED counts, and at most the second. */
        private const val AE_LOCK_MIN_MS = 300L
        private const val AE_LOCK_MAX_MS = 1_200L
        /** A tap's AF waits at least this for AE to react to the new region, and at most the second. */
        private const val AE_SETTLE_MIN_MS = 150L
        private const val AE_SETTLE_MAX_MS = 400L
        /** Half the tap region's side, of the active array's short side. */
        private const val TAP_REGION_HALF = 0.06f
        /** After opening, the scene's sweep waits this long for AE, and at most the second. */
        private const val SCENE_SWEEP_MIN_MS = 300L
        private const val SCENE_SWEEP_MAX_WAIT_MS = 1_500L
        private const val FULL_SWEEP_MAX_MS = 1_500L
        private const val REFINE_MAX_MS = 1_000L
        private const val STEP_TIMEOUT_MS = 300L
        /** Frames per sweep step; the last is scored, the lens has settled by then. The first step may move the lens across its whole range. */
        private const val COARSE_FRAMES = 2
        private const val FINE_FRAMES = 2
        private const val FIRST_STEP_EXTRA_FRAMES = 3
        /** A fine sweep covers this many fine steps either side, and follows an edge peak at most [MAX_FINE_EXTENSIONS] times. */
        private const val FINE_HALF_STEPS = 3
        private const val FINE_EXTENSION_STEPS = 2
        private const val MAX_FINE_EXTENSIONS = 2
        private const val SWEEP_PENDING_MAX = 8
        private const val MAX_ANALYSIS_WIDTH = 1920
        private const val YUV_MAX_IMAGES = 3
        /** Gyro: faster than this over [MOVE_WINDOW_NS] is turning the phone; slower than [STILL_RAD_PER_SEC] over [SETTLE_WINDOW_NS] is holding it. */
        private const val MOVING_RAD_PER_SEC = 0.15f
        private const val MOVE_WINDOW_NS = 150_000_000L
        private const val STILL_RAD_PER_SEC = 0.06f
        private const val SETTLE_WINDOW_NS = 500_000_000L
        /** A turn smaller than this at under [SMALL_EV_CHANGE] stops of exposure change is swept only finely. */
        private const val SMALL_TURN_RAD = 0.1f
        private const val SMALL_EV_CHANGE = 1.0
        private const val BURST_FOCUS_TOLERANCE = 0.05f
        private const val HOLD_SETTLE_MAX_MS = 300L
        /** Red and blue gain ratio per white balance step (about 4%, a few hundred kelvin near daylight). */
        private const val WB_STEP = 1.04
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

        private fun afStateName(af: Int?): String = when (af) {
            CaptureResult.CONTROL_AF_STATE_INACTIVE -> "INACTIVE"
            CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN -> "PASSIVE_SCAN"
            CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED -> "PASSIVE_FOCUSED"
            CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN -> "ACTIVE_SCAN"
            CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> "FOCUSED_LOCKED"
            CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED -> "NOT_FOCUSED_LOCKED"
            CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED -> "PASSIVE_UNFOCUSED"
            else -> "$af"
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
