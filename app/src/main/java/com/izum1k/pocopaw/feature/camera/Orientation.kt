package com.izum1k.pocopaw.feature.camera

import android.view.OrientationEventListener
import androidx.compose.animation.core.animateFloatAsState
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

private const val HYSTERESIS = 10
