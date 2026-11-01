package com.izum1k.pocopaw.feature.camera

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.lerp
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlin.math.PI
import kotlin.math.min
import kotlin.math.sin
import androidx.compose.material3.LoadingIndicator
import androidx.compose.material3.WavyProgressIndicatorDefaults
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.material3.MaterialShapes
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.ToggleButton
import androidx.compose.material3.ToggleButtonDefaults
import androidx.compose.material3.toShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.graphics.shapes.Morph
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.data.CameraSettings
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.ui.component.PillShape
import com.izum1k.pocopaw.ui.component.Reveal
import com.izum1k.pocopaw.ui.theme.MorphShape
import com.izum1k.pocopaw.ui.theme.YoinContainerShapes
import com.izum1k.pocopaw.ui.theme.YoinMotion
import com.izum1k.pocopaw.ui.theme.withTabularFigures
import kotlin.math.roundToInt

private const val CONTROL_ALPHA = 0.72f

@Composable
private fun controlContainer(): Color = MaterialTheme.colorScheme.surfaceContainer.copy(alpha = CONTROL_ALPHA)

@Composable
fun TopControls(
    settings: CameraSettings,
    hasFlash: Boolean,
    rotation: Float,
    onFlash: (FlashMode) -> Unit,
    onTimer: (TimerMode) -> Unit,
    onAspect: (AspectRatio) -> Unit,
    onGrid: (Boolean) -> Unit,
    onSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val haptics = LocalHapticFeedback.current
    Row(modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Row(
            Modifier
                .clip(PillShape)
                .background(controlContainer())
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            if (hasFlash) {
                QuickToggle(
                    caption = stringResource(R.string.control_flash),
                    value = stringResource(flashLabel(settings.flash)),
                    checked = settings.flash != FlashMode.OFF,
                    rotation = rotation,
                ) {
                    haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                    onFlash(FlashMode.entries[(settings.flash.ordinal + 1) % FlashMode.entries.size])
                }
            }
            QuickToggle(
                caption = stringResource(R.string.control_timer),
                value = if (settings.timer == TimerMode.OFF) stringResource(R.string.value_off) else stringResource(R.string.value_seconds, settings.timer.seconds),
                checked = settings.timer != TimerMode.OFF,
                rotation = rotation,
            ) {
                haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                onTimer(TimerMode.entries[(settings.timer.ordinal + 1) % TimerMode.entries.size])
            }
            QuickToggle(
                caption = stringResource(R.string.control_aspect),
                value = "${settings.aspect.width}:${settings.aspect.height}",
                checked = settings.aspect != AspectRatio.FOUR_THREE,
                rotation = rotation,
            ) {
                haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                onAspect(if (settings.aspect == AspectRatio.FOUR_THREE) AspectRatio.SIXTEEN_NINE else AspectRatio.FOUR_THREE)
            }
            QuickToggle(
                caption = stringResource(R.string.control_grid),
                value = stringResource(if (settings.grid) R.string.value_on else R.string.value_off),
                checked = settings.grid,
                rotation = rotation,
            ) {
                haptics.performHapticFeedback(if (settings.grid) HapticFeedbackType.ToggleOff else HapticFeedbackType.ToggleOn)
                onGrid(!settings.grid)
            }
        }
        Spacer(Modifier.weight(1f))
        FilledTonalIconButton(
            onClick = {
                haptics.performHapticFeedback(HapticFeedbackType.ContextClick)
                onSettings()
            },
            shapes = IconButtonDefaults.shapes(),
            modifier = Modifier.size(56.dp),
        ) {
            Icon(
                YoinSymbols.Settings,
                contentDescription = stringResource(R.string.settings_title),
                modifier = Modifier.graphicsLayer { rotationZ = rotation },
            )
        }
    }
}

@Composable
private fun QuickToggle(caption: String, value: String, checked: Boolean, rotation: Float, onClick: () -> Unit) {
    ToggleButton(
        checked = checked,
        onCheckedChange = { onClick() },
        // Concentric with the 56dp pill around it (4dp inset): pill at rest and checked alike.
        shapes = ToggleButtonDefaults.shapes(shape = PillShape, pressedShape = QuickTogglePressed, checkedShape = PillShape),
        colors = ToggleButtonDefaults.toggleButtonColors(containerColor = Color.Transparent),
        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 4.dp),
        modifier = Modifier.size(width = 60.dp, height = 48.dp).semantics { contentDescription = "$caption $value" },
    ) {
        Column(
            Modifier.graphicsLayer { rotationZ = rotation },
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(caption, style = MaterialTheme.typography.labelSmall, maxLines = 1)
            val effects = YoinMotion.defaultEffectsSpec<Float>()
            val fastEffects = YoinMotion.fastEffectsSpec<Float>()
            val spatial = YoinMotion.fastSpatialSpec<Float>()
            AnimatedContent(
                targetState = value,
                transitionSpec = { (fadeIn(effects) + scaleIn(spatial, 0.8f)) togetherWith fadeOut(fastEffects) },
                label = "toggleValue",
            ) { v -> Text(v, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold, maxLines = 1) }
        }
    }
}

private val QuickTogglePressed = RoundedCornerShape(14.dp)

/**
 * The seam between viewfinder and controls, in PiliPaw's wave (Yoin's WaveProgressBar): a quiet
 * straight line that swells into M3E's flowing wave while RAW frames are coming off the sensor.
 * A single frame takes ~100ms, so every capture swells all the way before it settles back.
 */
@Composable
fun ControlsWave(capturing: Boolean, modifier: Modifier = Modifier) {
    val base = WavyProgressIndicatorDefaults.linearIndicatorStroke
    val stroke = remember(base) { Stroke(width = base.width * 0.92f, cap = base.cap) }
    val amplitude = remember { Animatable(0f) }
    val current by rememberUpdatedState(capturing)
    val rise = YoinMotion.fastSpatialSpec<Float>()
    val settle = YoinMotion.slowSpatialSpec<Float>()
    LaunchedEffect(amplitude) {
        while (true) {
            snapshotFlow { current }.first { it }
            amplitude.animateTo(1f, rise)
            snapshotFlow { current }.first { !it }
            amplitude.animateTo(0f, settle)
        }
    }
    // M3E's indicator eases amplitude on its own clock and only while its progress moves; a
    // seam that must go flat on cue draws the same wave itself.
    var phase by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(amplitude) {
        var last = 0L
        snapshotFlow { amplitude.value > 0f }.collectLatest { moving ->
            last = 0L
            while (moving) {
                withFrameNanos { now ->
                    if (last != 0L) phase += (now - last) / 1e9f * WAVE_SPEED_PER_S
                    last = now
                }
            }
        }
    }
    val rest = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f)
    val active = MaterialTheme.colorScheme.primary
    val path = remember { Path() }
    Canvas(modifier.fillMaxWidth().height(WAVE_HEIGHT)) {
        val a = amplitude.value.coerceIn(0f, 1f)
        val width = stroke.width
        val half = width / 2f
        val mid = size.height / 2f
        val peak = (size.height - width) / 2f * a
        val wavelength = WAVE_LENGTH.toPx()
        val shift = phase * density
        path.rewind()
        var x = half
        path.moveTo(x, mid + peak * sin((x - shift) / wavelength * 2f * PI.toFloat()))
        val step = 2.dp.toPx()
        while (x < size.width - half) {
            x = min(x + step, size.width - half)
            path.lineTo(x, mid + peak * sin((x - shift) / wavelength * 2f * PI.toFloat()))
        }
        drawPath(path, lerp(rest, active, a), style = Stroke(width = width, cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}

val WAVE_HEIGHT = 14.dp
private val WAVE_LENGTH = 24.dp

/** dp per second the crests travel (PiliPaw's wave scrubber runs at 24, this one hurries). */
private const val WAVE_SPEED_PER_S = 48f

private fun flashLabel(mode: FlashMode): Int = when (mode) {
    FlashMode.OFF -> R.string.value_off
    FlashMode.AUTO -> R.string.value_auto
    FlashMode.ON -> R.string.value_on
}

/** One pill per back camera; the selected one shows the live zoom. */
@Composable
fun LensSelector(
    lenses: List<CameraInfo>,
    current: CameraInfo?,
    zoom: Float,
    rotation: Float,
    onSelect: (CameraInfo) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (lenses.size < 2 && (current == null || lenses.none { it.id == current.id })) return
    val haptics = LocalHapticFeedback.current
    Row(
        modifier
            .clip(PillShape)
            .background(controlContainer())
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        lenses.forEach { lens ->
            val selected = lens.id == current?.id
            val size by animateDpAsState(if (selected) 44.dp else 36.dp, YoinMotion.defaultSpatialSpec(), label = "lensSize")
            val container by animateColorAsState(
                if (selected) MaterialTheme.colorScheme.primary else Color.Transparent,
                YoinMotion.defaultEffectsSpec(),
                label = "lensContainer",
            )
            val content by animateColorAsState(
                if (selected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                YoinMotion.defaultEffectsSpec(),
                label = "lensContent",
            )
            Box(
                Modifier
                    .size(size)
                    .clip(CircleShape)
                    .background(container)
                    .clickable {
                        haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                        onSelect(lens)
                    },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    CameraFormat.zoom(if (selected) lens.relativeZoom * zoom else lens.relativeZoom),
                    style = MaterialTheme.typography.labelMedium.withTabularFigures(),
                    fontWeight = FontWeight.SemiBold,
                    color = content,
                    modifier = Modifier.graphicsLayer { rotationZ = rotation },
                )
            }
        }
    }
}

/** RAW size, live exposure and the RAW requests left before the device is reopened. */
@Composable
fun DevChip(info: CameraInfo?, remaining: Int, limit: Int, live: LiveExposure, lastRun: PipelineRun?, modifier: Modifier = Modifier) {
    if (info == null) return
    val low = remaining <= 5
    val remainingColor by animateColorAsState(
        if (low) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
        YoinMotion.defaultEffectsSpec(),
        label = "budgetColor",
    )
    val numbers = MaterialTheme.typography.labelMedium.withTabularFigures()
    val quiet = MaterialTheme.colorScheme.onSurfaceVariant
    // The last pipeline run gets a pill of its own: one row with it overflows a phone's width.
    Column(modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        DevPill {
            Text("RAW", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary, softWrap = false)
            Text("${info.rawSize.width}×${info.rawSize.height}", style = numbers, softWrap = false)
            live.iso?.let { Text("ISO $it", style = numbers, color = quiet, softWrap = false) }
            live.exposureNs?.let { Text(CameraFormat.shutter(it), style = numbers, color = quiet, softWrap = false) }
            Text("$remaining/$limit", style = numbers, color = remainingColor, softWrap = false)
        }
        Reveal(lastRun != null) {
            val run = lastRun ?: return@Reveal
            DevPill {
                Text(stringResource(R.string.dev_processed, run.frames, run.millis / 1000f), style = numbers, color = quiet, softWrap = false)
            }
        }
    }
}

@Composable
private fun DevPill(content: @Composable RowScope.() -> Unit) {
    Row(
        Modifier
            .clip(PillShape)
            .background(controlContainer())
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
        content = content,
    )
}

@Composable
fun EvChip(label: String, rotation: Float, modifier: Modifier = Modifier) {
    Row(
        modifier
            .graphicsLayer { rotationZ = rotation }
            .clip(PillShape)
            .background(controlContainer())
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text("EV", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
        Text(label, style = MaterialTheme.typography.labelMedium.withTabularFigures())
    }
}

@Composable
fun ModeBar(mode: CaptureMode, onMode: (CaptureMode) -> Unit, modifier: Modifier = Modifier) {
    val haptics = LocalHapticFeedback.current
    Row(modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
        CaptureMode.entries.forEach { m ->
            val selected = m == mode
            val container by animateColorAsState(
                if (selected) MaterialTheme.colorScheme.secondaryContainer else Color.Transparent,
                YoinMotion.defaultEffectsSpec(),
                label = "modeContainer",
            )
            val content by animateColorAsState(
                if (selected) MaterialTheme.colorScheme.onSecondaryContainer else MaterialTheme.colorScheme.onSurfaceVariant,
                YoinMotion.defaultEffectsSpec(),
                label = "modeContent",
            )
            Box(
                Modifier
                    .padding(horizontal = 4.dp)
                    .clip(PillShape)
                    .background(container)
                    .clickable {
                        if (!selected) {
                            haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                            onMode(m)
                        }
                    }
                    .padding(horizontal = 20.dp, vertical = 6.dp),
            ) {
                Text(
                    stringResource(if (m == CaptureMode.PHOTO) R.string.mode_photo else R.string.mode_pro),
                    style = MaterialTheme.typography.titleSmall,
                    color = content,
                )
            }
        }
    }
}

/** Manual ISO and shutter on the camera's own stops. */
@Composable
fun ProPanel(
    camera: CameraInfo?,
    iso: Int,
    exposureNs: Long,
    onIso: (Int) -> Unit,
    onShutter: (Long) -> Unit,
    modifier: Modifier = Modifier,
) {
    val isoStops = CameraViewModel.isoStops(camera)
    val shutterStops = CameraViewModel.shutterStops(camera)
    Column(
        modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
            .clip(YoinContainerShapes.Card)
            .background(controlContainer())
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        StopSlider(
            label = "ISO",
            value = iso.toString(),
            index = isoStops.indexOf(CameraViewModel.snapIso(iso, camera)),
            count = isoStops.size,
            onIndex = { onIso(isoStops[it]) },
        )
        StopSlider(
            label = stringResource(R.string.control_shutter),
            value = CameraFormat.shutter(exposureNs),
            index = shutterStops.indexOf(CameraViewModel.snapShutter(exposureNs, camera)),
            count = shutterStops.size,
            onIndex = { onShutter(shutterStops[it]) },
        )
    }
}

@Composable
private fun StopSlider(label: String, value: String, index: Int, count: Int, onIndex: (Int) -> Unit) {
    val haptics = LocalHapticFeedback.current
    var last by remember { mutableIntStateOf(index) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, softWrap = false, modifier = Modifier.width(60.dp))
        Text(
            value,
            style = MaterialTheme.typography.titleSmall.withTabularFigures(),
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.widthIn(min = 56.dp),
        )
        Slider(
            value = index.coerceAtLeast(0).toFloat(),
            onValueChange = { v ->
                val i = v.roundToInt().coerceIn(0, count - 1)
                if (i != last) {
                    last = i
                    haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                    onIndex(i)
                }
            },
            valueRange = 0f..(count - 1).coerceAtLeast(1).toFloat(),
            steps = (count - 2).coerceAtLeast(0),
            enabled = count > 1,
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
fun ShutterRow(
    last: Bitmap?,
    hasLast: Boolean,
    busy: Busy,
    countdown: Int?,
    canFlip: Boolean,
    rotation: Float,
    onThumbnail: () -> Unit,
    onShutter: () -> Unit,
    onFlip: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier.fillMaxWidth().padding(horizontal = 32.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Thumbnail(last, hasLast, rotation, onThumbnail)
        ShutterButton(busy, countdown, onShutter)
        FlipButton(canFlip && busy == Busy.IDLE, rotation, onFlip)
    }
}

@Composable
private fun Thumbnail(bitmap: Bitmap?, hasLast: Boolean, rotation: Float, onClick: () -> Unit) {
    val haptics = LocalHapticFeedback.current
    Box(
        Modifier
            .size(52.dp)
            .clip(MaterialShapes.Cookie9Sided.toShape())
            .background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .clickable(enabled = hasLast) {
                haptics.performHapticFeedback(HapticFeedbackType.ContextClick)
                onClick()
            },
        contentAlignment = Alignment.Center,
    ) {
        val spatial = YoinMotion.defaultSpatialSpec<Float>()
        val effects = YoinMotion.defaultEffectsSpec<Float>()
        val fastEffects = YoinMotion.fastEffectsSpec<Float>()
        AnimatedContent(
            targetState = bitmap,
            transitionSpec = { (scaleIn(spatial, 0.6f) + fadeIn(effects)) togetherWith fadeOut(fastEffects) },
            label = "thumbnail",
        ) { b ->
            if (b != null) {
                Image(b.asImageBitmap(), contentDescription = stringResource(R.string.action_open_last), contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            } else {
                Icon(
                    YoinSymbols.Image,
                    contentDescription = if (hasLast) stringResource(R.string.action_open_last) else null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.graphicsLayer { rotationZ = rotation },
                )
            }
        }
    }
}

@Composable
private fun ShutterButton(busy: Busy, countdown: Int?, onClick: () -> Unit) {
    val haptics = LocalHapticFeedback.current
    val description = stringResource(R.string.action_shutter)
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val working = busy == Busy.CAPTURING || busy == Busy.PROCESSING
    val scale by animateFloatAsState(if (pressed) 0.86f else 1f, YoinMotion.fastSpatialSpec(), label = "shutterScale")
    val morphProgress by animateFloatAsState(if (pressed || working) 1f else 0f, YoinMotion.defaultSpatialSpec(), label = "shutterMorph")
    val morph = remember { Morph(MaterialShapes.Circle, MaterialShapes.Cookie9Sided) }
    val fill by animateColorAsState(
        when {
            busy == Busy.COUNTDOWN -> MaterialTheme.colorScheme.tertiary
            working -> MaterialTheme.colorScheme.primaryContainer
            else -> MaterialTheme.colorScheme.primary
        },
        YoinMotion.defaultEffectsSpec(),
        label = "shutterFill",
    )
    Box(
        Modifier
            .size(76.dp)
            .clip(CircleShape)
            .background(Color.White.copy(alpha = 0.18f))
            .clickable(interactionSource = interaction, indication = null, enabled = !working) {
                haptics.performHapticFeedback(HapticFeedbackType.Confirm)
                onClick()
            }
            .semantics { contentDescription = description },
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .size(62.dp)
                .graphicsLayer {
                    scaleX = scale
                    scaleY = scale
                }
                .clip(MorphShape(morph, morphProgress))
                .background(fill),
            contentAlignment = Alignment.Center,
        ) {
            when {
                busy == Busy.PROCESSING -> LoadingIndicator(color = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(36.dp))
                countdown != null -> Text(
                    countdown.toString(),
                    style = MaterialTheme.typography.titleLarge.withTabularFigures(),
                    color = MaterialTheme.colorScheme.onTertiary,
                )
            }
        }
    }
}

@Composable
private fun FlipButton(enabled: Boolean, rotation: Float, onClick: () -> Unit) {
    val haptics = LocalHapticFeedback.current
    var flips by remember { mutableIntStateOf(0) }
    val spin by animateFloatAsState(flips * 180f, YoinMotion.defaultSpatialSpec(), label = "flipSpin")
    val alpha by animateFloatAsState(if (enabled) 1f else 0.38f, YoinMotion.defaultEffectsSpec(), label = "flipAlpha")
    FilledTonalIconButton(
        onClick = {
            haptics.performHapticFeedback(HapticFeedbackType.Confirm)
            flips++
            onClick()
        },
        enabled = enabled,
        shapes = IconButtonDefaults.shapes(),
        modifier = Modifier.size(52.dp).graphicsLayer { this.alpha = alpha },
    ) {
        Icon(
            YoinSymbols.Refresh,
            contentDescription = stringResource(R.string.action_flip),
            modifier = Modifier.graphicsLayer { rotationZ = rotation + spin },
        )
    }
}