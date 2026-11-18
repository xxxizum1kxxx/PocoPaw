package com.izum1k.pocopaw.feature.camera

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LoadingIndicator
import androidx.compose.material3.MaterialShapes
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.ToggleButton
import androidx.compose.material3.ToggleButtonDefaults
import androidx.compose.material3.WavyProgressIndicatorDefaults
import androidx.compose.material3.toShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.dp
import androidx.graphics.shapes.Morph
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.data.CameraSettings
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.ui.component.PillShape
import com.izum1k.pocopaw.ui.theme.MorphShape
import com.izum1k.pocopaw.ui.theme.YoinMotion
import com.izum1k.pocopaw.ui.theme.withTabularFigures
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlin.math.PI
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

private const val CONTROL_ALPHA = 0.72f

@Composable
private fun controlContainer(): Color = MaterialTheme.colorScheme.surfaceContainer.copy(alpha = CONTROL_ALPHA)

@Composable
fun TopControls(
    settings: CameraSettings,
    hasFlash: Boolean,
    flashRequired: Boolean,
    rotation: Float,
    onFlash: (FlashMode) -> Unit,
    onTimer: (TimerMode) -> Unit,
    onAspect: (AspectRatio) -> Unit,
    onGrid: (Boolean) -> Unit,
    onSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val haptics = LocalHapticFeedback.current
    Row(modifier.fillMaxWidth().padding(start = 24.dp, end = 24.dp, top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
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
                    value = flashValue(settings.flash, flashRequired),
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
private fun QuickToggle(caption: String, value: String, checked: Boolean, rotation: Float, onClick: () -> Unit) =
    QuickToggle(caption, AnnotatedString(value), checked, rotation, onClick)

@Composable
private fun QuickToggle(caption: String, value: AnnotatedString, checked: Boolean, rotation: Float, onClick: () -> Unit) {
    ToggleButton(
        checked = checked,
        onCheckedChange = { onClick() },
        shapes = ToggleButtonDefaults.shapes(shape = PillShape, pressedShape = QuickTogglePressed, checkedShape = PillShape),
        colors = ToggleButtonDefaults.toggleButtonColors(containerColor = Color.Transparent),
        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 4.dp),
        modifier = Modifier.size(width = 54.dp, height = 48.dp).semantics { contentDescription = "$caption ${value.text}" },
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
private const val WAVE_SPEED_PER_S = 48f

@Composable
private fun flashValue(mode: FlashMode, flashRequired: Boolean): AnnotatedString {
    var metered by remember { mutableStateOf(false) }
    LaunchedEffect(mode) {
        metered = false
        if (mode == FlashMode.AUTO) {
            delay(FLASH_METER_MS)
            metered = true
        }
    }
    val auto = stringResource(R.string.value_auto)
    val on = stringResource(R.string.value_on)
    val off = stringResource(R.string.value_off)
    return when {
        mode == FlashMode.OFF -> AnnotatedString(off)
        mode == FlashMode.ON -> AnnotatedString(on)
        !metered -> AnnotatedString(auto)
        else -> buildAnnotatedString {
            withStyle(SpanStyle(fontWeight = FontWeight.Black)) { append("A") }
            withStyle(SpanStyle(fontWeight = FontWeight.Normal)) { append("/" + if (flashRequired) on else off) }
        }
    }
}

private const val FLASH_METER_MS = 1_200L

/**
 * One pill per back camera; the selected one shows the live zoom. Tapping the selected main lens
 * (1×) calls [onToggleZoom] for its 2× step instead of showing the focal length.
 */
@Composable
fun LensSelector(
    lenses: List<CameraInfo>,
    current: CameraInfo?,
    zoom: Float,
    rotation: Float,
    onSelect: (CameraInfo) -> Unit,
    onToggleZoom: () -> Unit,
    modifier: Modifier = Modifier,
) {
    if (lenses.size < 2 && (current == null || lenses.none { it.id == current.id })) return
    val haptics = LocalHapticFeedback.current
    var focalTap by remember { mutableIntStateOf(0) }
    var showFocal by remember { mutableStateOf(false) }
    LaunchedEffect(focalTap) {
        if (focalTap == 0) return@LaunchedEffect
        showFocal = true
        delay(FOCAL_SHOW_MS)
        showFocal = false
    }
    val labelStyle = MaterialTheme.typography.labelMedium.withTabularFigures().copy(fontWeight = FontWeight.SemiBold)
    val measurer = rememberTextMeasurer()
    val density = LocalDensity.current
    val sizes = lenses.map { lens ->
        animateDpAsState(if (lens.id == current?.id) 40.dp else 34.dp, YoinMotion.defaultSpatialSpec(), label = "lensSize")
    }
    val restWidths = lenses.mapIndexed { i, lens ->
        val factor = if (lens.id == current?.id) lens.relativeZoom * zoom else lens.relativeZoom
        val text = with(density) { measurer.measure(CameraFormat.zoom(factor), labelStyle).size.width.toDp() }
        maxOf(sizes[i].value, text + LENS_LABEL_PADDING * 2)
    }
    val pillColor = controlContainer()
    Layout(
        modifier = modifier,
        content = {
            Box(Modifier.clip(PillShape).background(pillColor))
            lenses.forEachIndexed { i, lens ->
                val selected = lens.id == current?.id
                val size by sizes[i]
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
                val factor = if (selected) lens.relativeZoom * zoom else lens.relativeZoom
                val label = if (showFocal && selected) {
                    CameraFormat.equivalentFocal(lens.equivalentFocalMm * zoom)
                } else {
                    CameraFormat.zoom(factor)
                }
                Box(
                    Modifier
                        .height(size)
                        .widthIn(min = size)
                        .clip(PillShape)
                        .background(container)
                        .clickable {
                            haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                            if (selected && lens.relativeZoom == 1f) {
                                showFocal = false
                                onToggleZoom()
                            } else {
                                focalTap++
                                onSelect(lens)
                            }
                        }
                        .animateContentSize(YoinMotion.defaultSpatialSpec())
                        .padding(horizontal = LENS_LABEL_PADDING),
                    contentAlignment = Alignment.Center,
                ) {
                    val effects = YoinMotion.defaultEffectsSpec<Float>()
                    val fastEffects = YoinMotion.fastEffectsSpec<Float>()
                    val spatial = YoinMotion.fastSpatialSpec<Float>()
                    AnimatedContent(
                        targetState = label,
                        transitionSpec = {
                            if (initialState.endsWith("mm") != targetState.endsWith("mm")) {
                                (fadeIn(effects) + scaleIn(spatial, 0.8f)) togetherWith fadeOut(fastEffects)
                            } else {
                                EnterTransition.None togetherWith ExitTransition.None
                            }
                        },
                        label = "lensLabel",
                    ) { text ->
                        Text(
                            text,
                            style = labelStyle,
                            color = content,
                            softWrap = false,
                            modifier = Modifier.graphicsLayer { rotationZ = rotation },
                        )
                    }
                }
            }
        },
    ) { measurables, _ ->
        val pad = LENS_PILL_PADDING.roundToPx()
        val gap = LENS_GAP.roundToPx()
        val buttons = measurables.drop(1).map { it.measure(Constraints()) }
        val n = buttons.size
        val rest = restWidths.map { it.roundToPx() }
        val height = (buttons.maxOfOrNull { it.height } ?: 0) + 2 * pad
        val spacing = gap * (n - 1).coerceAtLeast(0) + 2 * pad
        val actual = buttons.sumOf { it.width } + spacing
        val laidOut = rest.sum() + spacing
        val shift = buttons.indices.sumOf { i ->
            val growLeft = if (n < 2) 0.5f else 1f - i / (n - 1f)
            ((buttons[i].width - rest[i]) * growLeft).roundToInt()
        }
        val pill = measurables[0].measure(Constraints.fixed(actual, height))
        layout(laidOut, height) {
            pill.place(-shift, 0)
            var x = -shift + pad
            buttons.forEach { b ->
                b.place(x, (height - b.height) / 2)
                x += b.width + gap
            }
        }
    }
}

private val LENS_LABEL_PADDING = 6.dp
private val LENS_PILL_PADDING = 4.dp
private val LENS_GAP = 4.dp
private const val FOCAL_SHOW_MS = 5_000L

@Composable
fun EvChip(label: String, modifier: Modifier = Modifier, tag: String = "EV", onClick: (() -> Unit)? = null) {
    val haptics = LocalHapticFeedback.current
    Row(
        modifier
            .clip(PillShape)
            .background(controlContainer())
            .then(
                if (onClick == null) {
                    Modifier
                } else {
                    Modifier.clickable {
                        haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                        onClick()
                    }
                },
            )
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(tag, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
        Text(label, style = MaterialTheme.typography.labelMedium.withTabularFigures())
    }
}

@Composable
fun ShutterRow(
    last: Bitmap?,
    hasLast: Boolean,
    awaitingGallery: Boolean = false,
    busy: Busy,
    processing: Boolean,
    progress: Float,
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
        Thumbnail(last, hasLast, awaitingGallery, rotation, onThumbnail)
        ShutterButton(busy, processing, progress, countdown, rotation, onShutter)
        FlipButton(canFlip && busy == Busy.IDLE, rotation, onFlip)
    }
}

@Composable
private fun Thumbnail(
    bitmap: Bitmap?,
    hasLast: Boolean,
    awaitingGallery: Boolean,
    rotation: Float,
    onClick: () -> Unit,
) {
    val haptics = LocalHapticFeedback.current
    val description = stringResource(R.string.action_open_last)
    Box(
        Modifier
            .size(52.dp)
            .clip(MaterialShapes.Cookie9Sided.toShape())
            .background(MaterialTheme.colorScheme.surfaceContainerHigh)
            .clickable {
                haptics.performHapticFeedback(HapticFeedbackType.ContextClick)
                onClick()
            }
            .semantics { contentDescription = description },
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
                Image(b.asImageBitmap(), contentDescription = description, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            } else {
                Icon(
                    YoinSymbols.Image,
                    contentDescription = if (hasLast) description else null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.graphicsLayer { rotationZ = rotation },
                )
            }
        }
        if (awaitingGallery) {
            Box(
                Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.45f)),
                contentAlignment = Alignment.Center,
            ) {
                LoadingIndicator(color = Color.White, modifier = Modifier.size(24.dp))
            }
        }
    }
}

@Composable
internal fun ShutterButton(
    busy: Busy,
    processing: Boolean = false,
    progress: Float = 0f,
    countdown: Int? = null,
    rotation: Float = 0f,
    onClick: () -> Unit,
) {
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
    val ringProgress by animateFloatAsState(
        if (busy == Busy.IDLE && processing) progress.coerceIn(0f, 1f) else 0f,
        YoinMotion.defaultSpatialSpec(),
        label = "shutterProgress",
    )
    val arcColor = MaterialTheme.colorScheme.primary
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
        if (busy == Busy.IDLE && processing) {
            Canvas(Modifier.fillMaxSize()) {
                val stroke = 3.dp.toPx()
                drawArc(
                    color = arcColor,
                    startAngle = -90f,
                    sweepAngle = 360f * ringProgress,
                    useCenter = false,
                    topLeft = Offset(stroke / 2f, stroke / 2f),
                    size = Size(size.width - stroke, size.height - stroke),
                    style = Stroke(width = stroke, cap = StrokeCap.Round),
                )
            }
        }
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
                    modifier = Modifier.graphicsLayer { rotationZ = rotation },
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
