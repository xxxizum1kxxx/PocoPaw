package com.izum1k.pocopaw.feature.camera

import android.graphics.RectF
import android.graphics.SurfaceTexture
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import android.view.TextureView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LoadingIndicator
import androidx.compose.material3.MaterialShapes
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import kotlinx.coroutines.delay
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.layout.layout
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalViewConfiguration
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.graphics.shapes.Morph
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.data.GridStyle
import com.izum1k.pocopaw.ui.theme.MorphShape
import com.izum1k.pocopaw.ui.theme.YoinContainerShapes
import com.izum1k.pocopaw.ui.theme.YoinMotion
import com.izum1k.pocopaw.ui.theme.withTabularFigures
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt

/** One side's vertical swipe: the value and its range in steps, and the readout beside the gauge. */
data class SwipeDial(val value: Int, val min: Int, val max: Int, val label: String)

/** As the user holds the phone: the left half's swipe sets EV, the right half's white balance. */
enum class SwipeSide { EV, WB }

@Composable
fun Viewfinder(
    frame: AspectFrame,
    grid: Boolean,
    gridStyle: GridStyle,
    focus: FocusPoint?,
    focusLocked: Boolean,
    /** The tap lock holds until released rather than timing out. */
    focusHeld: Boolean,
    /** Faces in normalised viewfinder (texture) coordinates, like [focus]. */
    faces: List<RectF>,
    evLabel: String?,
    countdown: Int?,
    shutterTick: Int,
    cameraState: CameraController.State,
    live: Boolean,
    processing: Boolean,
    processingProgress: () -> Float,
    /** Null where that side's swipe does nothing. */
    ev: SwipeDial?,
    wb: SwipeDial?,
    controlRotation: Float,
    onSurfaceAvailable: (SurfaceTexture) -> Unit,
    onSurfaceDestroyed: () -> Unit,
    /** Normalised position, and whether it landed on the current focus ring. */
    onTap: (Float, Float, Boolean) -> Unit,
    onZoom: (Float) -> Unit,
    onEvSteps: (Int) -> Unit,
    onWbSteps: (Int) -> Unit,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
    topOverlay: @Composable BoxScope.() -> Unit = {},
    bottomOverlay: @Composable BoxScope.() -> Unit = {},
    /** Over everything, edge to edge of the frame. */
    cover: @Composable BoxScope.() -> Unit = {},
) {
    // The iris opens on the new camera's first frame, not on the old one's last.
    var fresh by remember { mutableStateOf(false) }
    LaunchedEffect(live) { if (!live) fresh = false }
    val irisOpen = YoinMotion.slowSpatialSpec<Float>()
    val irisShut = YoinMotion.fastSpatialSpec<Float>()
    LaunchedEffect(live && fresh) {
        if (live && fresh) frame.iris.animateTo(1f, irisOpen) else frame.iris.animateTo(0f, irisShut)
    }
    var swiping by remember { mutableStateOf<SwipeSide?>(null) }
    var lingering by remember { mutableStateOf<SwipeSide?>(null) }
    LaunchedEffect(swiping) {
        if (swiping != null) {
            lingering = swiping
        } else {
            delay(GAUGE_LINGER_MS)
            lingering = null
        }
    }
    BoxWithConstraints(
        modifier
            .fillMaxWidth()
            .layout { measurable, constraints ->
                val width = constraints.maxWidth
                val height = (width * frame.heightRatio).roundToInt()
                val placeable = measurable.measure(Constraints.fixed(width, height))
                layout(width, height) { placeable.place(0, 0) }
            }
            .clip(YoinContainerShapes.Panel)
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .requiredSize(maxWidth, maxWidth * SENSOR_RATIO)
                .graphicsLayer {
                    scaleX = frame.scale.value
                    scaleY = frame.scale.value
                }
                .aspectSeam(frame),
        ) {
            PreviewTexture(onSurfaceAvailable, onSurfaceDestroyed, onFrame = { if (live && !fresh) fresh = true }, Modifier.fillMaxSize())
            Box(
                Modifier
                    .fillMaxSize()
                    .viewfinderGestures(focus, ev != null, wb != null, controlRotation, onTap, onZoom, onEvSteps, onWbSteps) { swiping = it },
            )
            FaceBoxes(if (focus == null) faces else emptyList(), focusLocked)
            FocusRing(focus, focusLocked, focusHeld, evLabel, controlRotation)
        }

        if (grid) GridOverlay(gridStyle, Modifier.framedBand(frame))

        val gaugeSide = swiping ?: lingering
        SwipeGauge(gaugeSide, if (gaugeSide == SwipeSide.WB) wb else ev, swiping != null, controlRotation)

        AnimatedVisibility(
            visible = cameraState is CameraController.State.Opening,
            enter = fadeIn(YoinMotion.defaultEffectsSpec()),
            exit = fadeOut(YoinMotion.fastEffectsSpec()),
        ) { LoadingIndicator() }

        ProcessingVeil(processing, processingProgress)

        if (cameraState is CameraController.State.Error) ErrorPanel(cameraState.reason, onRetry)

        val spatial = YoinMotion.defaultSpatialSpec<Float>()
        val fastSpatial = YoinMotion.fastSpatialSpec<Float>()
        val effects = YoinMotion.defaultEffectsSpec<Float>()
        val fastEffects = YoinMotion.fastEffectsSpec<Float>()
        AnimatedContent(
            targetState = countdown,
            transitionSpec = {
                (scaleIn(spatial, initialScale = 1.6f) + fadeIn(effects)) togetherWith
                    (scaleOut(fastSpatial, targetScale = 0.6f) + fadeOut(fastEffects))
            },
            label = "countdown",
        ) { n ->
            if (n != null) {
                Text(
                    n.toString(),
                    style = MaterialTheme.typography.displayLarge.withTabularFigures(),
                    color = Color.White,
                    modifier = Modifier.graphicsLayer { rotationZ = controlRotation },
                )
            }
        }

        ShutterFlash(shutterTick)
        Box(Modifier.fillMaxSize().padding(12.dp), content = topOverlay)
        Box(Modifier.fillMaxSize().padding(12.dp), content = bottomOverlay)
        Box(Modifier.fillMaxSize(), content = cover)
    }
}

/** The part of the viewfinder the band shows, centred. */
private fun Modifier.framedBand(frame: AspectFrame): Modifier = layout { measurable, constraints ->
    val width = constraints.maxWidth
    val picture = width * SENSOR_RATIO
    val w = min(width.toFloat(), width * frame.bandWidth * frame.scale.value).roundToInt()
    val h = min(constraints.maxHeight.toFloat(), picture * frame.bandHeight * frame.scale.value).roundToInt()
    val placeable = measurable.measure(Constraints.fixed(w, h))
    layout(w, h) { placeable.place(0, 0) }
}

@Composable
private fun PreviewTexture(
    onSurfaceAvailable: (SurfaceTexture) -> Unit,
    onSurfaceDestroyed: () -> Unit,
    onFrame: () -> Unit,
    modifier: Modifier,
) {
    val available by rememberUpdatedState(onSurfaceAvailable)
    val destroyed by rememberUpdatedState(onSurfaceDestroyed)
    val frame by rememberUpdatedState(onFrame)
    AndroidView(
        modifier = modifier,
        factory = { context ->
            TextureView(context).apply {
                surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) = available(surface)
                    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) = Unit
                    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                        destroyed()
                        return true
                    }
                    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = frame()
                }
            }
        },
    )
}

/**
 * One gesture handler so tap, pinch and swipes never fight: two fingers zoom; a swipe along the
 * user's up/down axis (the phone as held, from [controlRotation]) past touch slop steps EV on the
 * user's left half and white balance on the right, up being more; anything else that lifts is a
 * focus tap, flagged when it lands within [RING_HIT] of [focus]. [onSwipe] reports the side being
 * swiped, then null.
 */
@Composable
private fun Modifier.viewfinderGestures(
    focus: FocusPoint?,
    evEnabled: Boolean,
    wbEnabled: Boolean,
    controlRotation: Float,
    onTap: (Float, Float, Boolean) -> Unit,
    onZoom: (Float) -> Unit,
    onEvSteps: (Int) -> Unit,
    onWbSteps: (Int) -> Unit,
    onSwipe: (SwipeSide?) -> Unit,
): Modifier {
    val tap by rememberUpdatedState(onTap)
    val zoom by rememberUpdatedState(onZoom)
    val ev by rememberUpdatedState(onEvSteps)
    val wb by rememberUpdatedState(onWbSteps)
    val swipe by rememberUpdatedState(onSwipe)
    val evOn by rememberUpdatedState(evEnabled)
    val wbOn by rememberUpdatedState(wbEnabled)
    val rotation by rememberUpdatedState(controlRotation)
    val ring by rememberUpdatedState(focus)
    val ringHitPx = with(LocalDensity.current) { RING_HIT.toPx() }
    val slop = LocalViewConfiguration.current.touchSlop
    val stepPx = with(LocalDensity.current) { 28.dp.toPx() }
    return pointerInput(Unit) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false)
            val up = userUp(rotation)
            val right = Offset(-up.y, up.x)
            var multi = false
            var dragging = false
            var side: SwipeSide? = null
            var travel = Offset.Zero
            var accum = 0f
            do {
                val event = awaitPointerEvent()
                val pressed = event.changes.count { it.pressed }
                if (pressed >= 2) {
                    multi = true
                    val z = event.calculateZoom()
                    if (z != 1f) zoom(z)
                    event.changes.forEach { it.consume() }
                } else if (!multi) {
                    val change = event.changes.first()
                    travel += change.positionChange()
                    val along = travel.dot(up)
                    if (!dragging && abs(along) > slop && abs(along) > abs(travel.dot(right))) {
                        dragging = true
                        val fromCentre = down.position - Offset(size.width / 2f, size.height / 2f)
                        side = (if (fromCentre.dot(right) < 0) SwipeSide.EV else SwipeSide.WB)
                            .takeIf { if (it == SwipeSide.EV) evOn else wbOn }
                        swipe(side)
                    }
                    if (dragging && side != null) {
                        accum += change.positionChange().dot(up)
                        val steps = (accum / stepPx).toInt()
                        if (steps != 0) {
                            if (side == SwipeSide.EV) ev(steps) else wb(steps)
                            accum -= steps * stepPx
                        }
                        change.consume()
                    }
                }
            } while (event.changes.any { it.pressed })
            if (dragging) swipe(null)
            if (!multi && !dragging && travel.getDistance() < slop) {
                val onRing = ring?.let { f ->
                    (down.position - Offset(f.x * size.width, f.y * size.height)).getDistance() < ringHitPx
                } ?: false
                tap(down.position.x / size.width, down.position.y / size.height, onRing)
            }
        }
    }
}

private val RING_HIT = 56.dp

private fun Offset.dot(o: Offset) = x * o.x + y * o.y

/** The user's up in screen coordinates, from the controls' counter-rotation, snapped to a quarter turn. */
private fun userUp(controlRotation: Float): Offset = when (Math.floorMod((controlRotation / 90f).roundToInt(), 4)) {
    0 -> Offset(0f, -1f)
    1 -> Offset(1f, 0f)
    2 -> Offset(0f, 1f)
    else -> Offset(-1f, 0f)
}

private const val GAUGE_LINGER_MS = 1_200L
private val GAUGE_HALF = 72.dp
private val GAUGE_EDGE = 28.dp
private val WARM = Color(0xFFFFB74D)
private val COOL = Color(0xFF64B5F6)

/**
 * A short vertical scale on the swiped side, upright for the user: centre notch at 0, the value's
 * dot (up is more) and its readout towards the middle. White balance runs cool to warm.
 */
@Composable
private fun SwipeGauge(side: SwipeSide?, dial: SwipeDial?, active: Boolean, controlRotation: Float) {
    val alpha by animateFloatAsState(if (side != null && dial != null && active) 1f else if (side != null && dial != null) 0.6f else 0f, YoinMotion.defaultEffectsSpec(), label = "gaugeAlpha")
    var shown by remember { mutableStateOf<Pair<SwipeSide, SwipeDial>?>(null) }
    if (side != null && dial != null) shown = side to dial
    val (s, d) = shown ?: return
    val measurer = rememberTextMeasurer()
    val textStyle = MaterialTheme.typography.labelLarge.withTabularFigures().copy(color = Color.White)
    val quarter = Math.floorMod((controlRotation / 90f).roundToInt(), 2) == 1
    Canvas(Modifier.fillMaxSize().graphicsLayer { this.alpha = alpha }) {
        val w = if (quarter) size.height else size.width
        val h = if (quarter) size.width else size.height
        rotate(controlRotation, center) {
            translate(center.x - w / 2, center.y - h / 2) {
                val half = GAUGE_HALF.toPx()
                val x = if (s == SwipeSide.EV) GAUGE_EDGE.toPx() else w - GAUGE_EDGE.toPx()
                val mid = h / 2
                val track = 3.dp.toPx()
                val brush = if (s == SwipeSide.WB) Brush.verticalGradient(listOf(WARM, COOL), mid - half, mid + half) else SolidColor(Color.White.copy(alpha = 0.7f))
                drawLine(Color.Black.copy(alpha = 0.35f), Offset(x, mid - half), Offset(x, mid + half), track + 2.dp.toPx(), StrokeCap.Round)
                drawLine(brush, Offset(x, mid - half), Offset(x, mid + half), track, StrokeCap.Round)
                val notch = 6.dp.toPx()
                drawLine(Color.White, Offset(x - notch, mid), Offset(x + notch, mid), 1.5.dp.toPx(), StrokeCap.Round)
                val span = maxOf(abs(d.min), abs(d.max), 1)
                val y = mid - d.value.toFloat() / span * half
                drawCircle(Color.Black.copy(alpha = 0.35f), 7.dp.toPx(), Offset(x, y))
                drawCircle(Color.White, 5.5.dp.toPx(), Offset(x, y))
                val text = measurer.measure(d.label, textStyle)
                val gap = 14.dp.toPx()
                val pad = 6.dp.toPx()
                val tx = if (s == SwipeSide.EV) x + gap else x - gap - text.size.width
                val ty = (y - text.size.height / 2f).coerceIn(0f, h - text.size.height)
                drawRoundRect(
                    Color.Black.copy(alpha = 0.45f),
                    topLeft = Offset(tx - pad, ty - pad / 2),
                    size = Size(text.size.width + 2 * pad, text.size.height + pad),
                    cornerRadius = CornerRadius(text.size.height.toFloat()),
                )
                drawText(text, topLeft = Offset(tx, ty))
            }
        }
    }
}

/** [held]: the lock no longer times out, shown by a label under the ring. */
@Composable
private fun BoxScope.FocusRing(focus: FocusPoint?, locked: Boolean, held: Boolean, evLabel: String?, controlRotation: Float) {
    if (focus == null) return
    val morph = remember { Morph(MaterialShapes.Cookie9Sided, MaterialShapes.Circle) }
    val progress by animateFloatAsState(if (locked) 1f else 0f, YoinMotion.defaultSpatialSpec(), label = "focusMorph")
    val scale = remember(focus.id) { Animatable(1.5f) }
    val settle = YoinMotion.spatialSpring<Float>()
    LaunchedEffect(focus.id) { scale.animateTo(1f, settle) }
    val color = if (locked) MaterialTheme.colorScheme.primary else Color.White
    val ringSize = 72.dp
    val density = LocalDensity.current
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val half = with(density) { (ringSize / 2).roundToPx() }
        val x = with(density) { (maxWidth * focus.x).roundToPx() } - half
        val y = with(density) { (maxHeight * focus.y).roundToPx() } - half
        Row(Modifier.offset { IntOffset(x, y) }, verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(ringSize)
                    .graphicsLayer {
                        scaleX = scale.value
                        scaleY = scale.value
                    }
                    .border(2.dp, color, MorphShape(morph, progress)),
            )
            if (evLabel != null) {
                Spacer(Modifier.width(8.dp))
                Text(
                    evLabel,
                    style = MaterialTheme.typography.labelLarge.withTabularFigures(),
                    color = color,
                    modifier = Modifier.graphicsLayer { rotationZ = controlRotation },
                )
            }
        }
        val labelAlpha by animateFloatAsState(if (held) 1f else 0f, YoinMotion.defaultEffectsSpec(), label = "holdLabel")
        val ringBottom = y + 2 * half + with(density) { 8.dp.roundToPx() }
        Box(
            Modifier
                .offset { IntOffset(x + half, ringBottom) }
                .layout { measurable, constraints ->
                    val p = measurable.measure(constraints.copy(minWidth = 0))
                    layout(p.width, p.height) { p.place(-p.width / 2, 0) }
                }
                .graphicsLayer {
                    alpha = labelAlpha
                    rotationZ = controlRotation
                }
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary)
                .padding(horizontal = 8.dp, vertical = 2.dp),
        ) {
            Text(
                stringResource(R.string.focus_held),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onPrimary,
                softWrap = false,
            )
        }
    }
}

/** A rounded frame per face; the one 3A follows is the largest, and turns primary once focus holds. */
@Composable
private fun FaceBoxes(faces: List<RectF>, focusLocked: Boolean) {
    val alpha by animateFloatAsState(if (faces.isEmpty()) 0f else 1f, YoinMotion.defaultEffectsSpec(), label = "faceAlpha")
    var shown by remember { mutableStateOf(faces) }
    if (faces.isNotEmpty()) shown = faces
    val primary = MaterialTheme.colorScheme.primary
    val lead = shown.maxByOrNull { it.width() * it.height() }
    Canvas(Modifier.fillMaxSize().graphicsLayer { this.alpha = alpha }) {
        val stroke = Stroke(2.dp.toPx())
        val corner = CornerRadius(12.dp.toPx())
        shown.forEach { f ->
            val color = if (f === lead && focusLocked) primary else Color.White.copy(alpha = 0.8f)
            drawRoundRect(
                color,
                topLeft = Offset(f.left * size.width, f.top * size.height),
                size = Size(f.width() * size.width, f.height() * size.height),
                cornerRadius = corner,
                style = stroke,
            )
        }
    }
}

@Composable
private fun GridOverlay(style: GridStyle, modifier: Modifier) {
    val lines = when (style) {
        GridStyle.THIRDS -> listOf(1f / 3f, 2f / 3f)
        GridStyle.GOLDEN -> listOf(0.382f, 0.618f)
        GridStyle.SQUARE -> listOf(0.25f, 0.5f, 0.75f)
    }
    Canvas(modifier) {
        val stroke = 1.dp.toPx()
        val c = Color.White.copy(alpha = 0.35f)
        lines.forEach { f ->
            drawLine(c, Offset(size.width * f, 0f), Offset(size.width * f, size.height), stroke)
            drawLine(c, Offset(0f, size.height * f), Offset(size.width, size.height * f), stroke)
        }
    }
}

/**
 * Over the viewfinder while the pipeline merges the shot. A plain dimming for now: the look is
 * still to be chosen, and [progress] (0..1 from the pipeline) is here for it.
 */
@Composable
private fun ProcessingVeil(visible: Boolean, @Suppress("UNUSED_PARAMETER") progress: () -> Float) {
    val alpha by animateFloatAsState(if (visible) 0.45f else 0f, YoinMotion.defaultEffectsSpec(), label = "processingVeil")
    Box(Modifier.fillMaxSize().graphicsLayer { this.alpha = alpha }.background(Color.Black))
}

@Composable
private fun ShutterFlash(tick: Int) {
    val alpha = remember { Animatable(0f) }
    val fade = YoinMotion.slowEffectsSpec<Float>()
    LaunchedEffect(tick) {
        if (tick == 0) return@LaunchedEffect
        alpha.snapTo(0.85f)
        alpha.animateTo(0f, fade)
    }
    Box(Modifier.fillMaxSize().graphicsLayer { this.alpha = alpha.value }.background(Color.Black))
}

@Composable
private fun ErrorPanel(reason: CameraController.Reason, onRetry: () -> Unit) {
    Column(
        Modifier
            .padding(32.dp)
            .clip(YoinContainerShapes.Card)
            .background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            stringResource(errorText(reason)),
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center,
        )
        Button(onClick = onRetry, shapes = ButtonDefaults.shapes()) { Text(stringResource(R.string.action_retry)) }
    }
}

private fun errorText(reason: CameraController.Reason): Int = when (reason) {
    CameraController.Reason.IN_USE, CameraController.Reason.MAX_IN_USE -> R.string.error_camera_in_use
    CameraController.Reason.DISABLED -> R.string.error_camera_disabled
    CameraController.Reason.SERVICE, CameraController.Reason.DISCONNECTED -> R.string.error_camera_service
    CameraController.Reason.CONFIGURATION -> R.string.error_camera_configuration
    else -> R.string.error_camera_device
}