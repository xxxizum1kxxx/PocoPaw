package com.izum1k.pocopaw.feature.camera

import android.view.OrientationEventListener
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.layout
import androidx.compose.ui.unit.Constraints
import kotlin.math.roundToInt
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.izum1k.pocopaw.ui.theme.YoinMotion
import kotlin.math.abs

/**
 * The device's rotation from natural portrait, clockwise, snapped to 0/90/180/270 with a margin
 * so it does not flicker near the diagonals. The activity stays portrait; controls counter-rotate.
 */
@Composable
fun rememberDeviceRotation(): State<Int> {
    val context = LocalContext.current
    val rotation = remember { mutableIntStateOf(0) }
    DisposableEffect(context) {
        val listener = object : OrientationEventListener(context) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation == ORIENTATION_UNKNOWN) return
                val current = rotation.intValue
                val diff = abs(((orientation - current + 540) % 360) - 180)
                if (diff > 45 + HYSTERESIS) rotation.intValue = ((orientation + 45) / 90 * 90) % 360
            }
        }
        if (listener.canDetectOrientation()) listener.enable()
        onDispose { listener.disable() }
    }
    return rotation
}

/** Counter-rotation for controls, taking the short way round (270 → 0 turns 90°, not 270°). */
@Composable
fun animatedControlRotation(deviceRotation: Int): Float {
    var target by remember { mutableFloatStateOf(-deviceRotation.toFloat()) }
    LaunchedEffect(deviceRotation) {
        var delta = (-deviceRotation.toFloat() - target) % 360f
        if (delta > 180f) delta -= 360f
        if (delta < -180f) delta += 360f
        target += delta
    }
    val angle by animateFloatAsState(target, YoinMotion.defaultSpatialSpec(), label = "controlRotation")
    return angle
}

/**
 * Lays [content] out upright for the user: in a box turned by [rotation] whose sides swap on a
 * quarter turn, so its corners are the user's corners rather than the portrait screen's.
 */
@Composable
fun UprightBox(rotation: Float, modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) {
    val swap = Math.floorMod((rotation / 90f).roundToInt(), 2) == 1
    Box(
        modifier.layout { measurable, constraints ->
            val w = constraints.maxWidth
            val h = constraints.maxHeight
            val pw = if (swap) h else w
            val ph = if (swap) w else h
            val placeable = measurable.measure(Constraints.fixed(pw, ph))
            layout(w, h) { placeable.placeWithLayer((w - pw) / 2, (h - ph) / 2) { rotationZ = rotation } }
        },
    ) {
        Box(Modifier.fillMaxSize(), content = content)
    }
}

private const val HYSTERESIS = 10
