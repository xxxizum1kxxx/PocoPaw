package com.izum1k.pocopaw.feature.camera

import android.graphics.SurfaceTexture
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
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
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

@Composable
fun Viewfinder(
    frame: AspectFrame,
    grid: Boolean,
    gridStyle: GridStyle,
    focus: FocusPoint?,
    focusLocked: Boolean,
    evLabel: String?,
    countdown: Int?,
    shutterTick: Int,
    cameraState: CameraController.State,
    evDragEnabled: Boolean,
    controlRotation: Float,
    onSurfaceAvailable: (SurfaceTexture) -> Unit,
    onSurfaceDestroyed: () -> Unit,
    onTap: (Float, Float) -> Unit,
    onZoom: (Float) -> Unit,
    onEvSteps: (Int) -> Unit,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
    topOverlay: @Composable BoxScope.() -> Unit = {},
    bottomOverlay: @Composable BoxScope.() -> Unit = {},
) {
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
            PreviewTexture(onSurfaceAvailable, onSurfaceDestroyed, Modifier.fillMaxSize())
            Box(
                Modifier
                    .fillMaxSize()
                    .viewfinderGestures(evDragEnabled, onTap, onZoom, onEvSteps),
            )
            FocusRing(focus, focusLocked, evLabel, controlRotation)
        }

        if (grid) GridOverlay(gridStyle, Modifier.framedBand(frame))

        val opening = cameraState is CameraController.State.Opening || cameraState is CameraController.State.Closed
        val scrim by animateFloatAsState(if (opening) 0.65f else 0f, YoinMotion.defaultEffectsSpec(), label = "openingScrim")
        Box(Modifier.fillMaxSize().graphicsLayer { alpha = scrim }.background(Color.Black))
        AnimatedVisibility(
            visible = cameraState is CameraController.State.Opening,
            enter = fadeIn(YoinMotion.defaultEffectsSpec()),
            exit = fadeOut(YoinMotion.fastEffectsSpec()),
        ) { LoadingIndicator() }

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
    modifier: Modifier,
) {
    val available by rememberUpdatedState(onSurfaceAvailable)
    val destroyed by rememberUpdatedState(onSurfaceDestroyed)
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
                    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
                }
            }
        },
    )
}

/**
 * One gesture handler so tap, pinch and vertical drag never fight: two fingers zoom, a vertical
 * drag past touch slop steps exposure compensation, anything else that lifts is a focus tap.
 */
@Composable
private fun Modifier.viewfinderGestures(
    evDragEnabled: Boolean,
    onTap: (Float, Float) -> Unit,
    onZoom: (Float) -> Unit,
    onEvSteps: (Int) -> Unit,
): Modifier {
    val tap by rememberUpdatedState(onTap)
    val zoom by rememberUpdatedState(onZoom)
    val ev by rememberUpdatedState(onEvSteps)
    val evEnabled by rememberUpdatedState(evDragEnabled)
    val slop = LocalViewConfiguration.current.touchSlop
    val stepPx = with(LocalDensity.current) { 28.dp.toPx() }
    return pointerInput(Unit) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false)
            var multi = false
            var dragging = false
            var travel = Offset.Zero
            var evAccum = 0f
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
                    if (!dragging && abs(travel.y) > slop && abs(travel.y) > abs(travel.x)) dragging = true
                    if (dragging && evEnabled) {
                        evAccum -= change.positionChange().y
                        val steps = (evAccum / stepPx).toInt()
                        if (steps != 0) {
                            ev(steps)
                            evAccum -= steps * stepPx
                        }
                        change.consume()
                    }
                }
            } while (event.changes.any { it.pressed })
            if (!multi && !dragging && travel.getDistance() < slop) {
                tap(down.position.x / size.width, down.position.y / size.height)
            }
        }
    }
}

@Composable
private fun BoxScope.FocusRing(focus: FocusPoint?, locked: Boolean, evLabel: String?, controlRotation: Float) {
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