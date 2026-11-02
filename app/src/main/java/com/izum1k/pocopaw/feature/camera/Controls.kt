package com.izum1k.pocopaw.feature.camera

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.EnterTransition
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
import kotlin.math.abs
import kotlin.math.log2
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import androidx.compose.material3.LoadingIndicator
import androidx.compose.material3.WavyProgressIndicatorDefaults
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.material3.MaterialShapes
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.ui.geometry.Offset
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.ui.input.pointer.pointerInput
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
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.layout.layout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import kotlinx.coroutines.delay
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.graphics.shapes.Morph
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.data.BurstFrames
import com.izum1k.pocopaw.data.CameraSettings
import com.izum1k.pocopaw.data.LookStyle
import com.izum1k.pocopaw.ui.component.ConnectedChoice
import com.izum1k.pocopaw.camera.ManualExposure
import com.izum1k.pocopaw.data.FlashMode
import com.izum1k.pocopaw.data.OutputFormat
import com.izum1k.pocopaw.data.Processing
import com.izum1k.pocopaw.data.SceneMode
import com.izum1k.pocopaw.data.TimerMode
import com.izum1k.pocopaw.ui.component.PillShape
import com.izum1k.pocopaw.ui.component.Reveal
import com.izum1k.pocopaw.ui.theme.MorphShape
import com.izum1k.pocopaw.ui.theme.YoinContainerShapes
import com.izum1k.pocopaw.ui.theme.YoinMotion
import com.izum1k.pocopaw.ui.theme.withTabularFigures
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
    quickOpen: Boolean,
    onQuick: (Boolean) -> Unit,
    onSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val haptics = LocalHapticFeedback.current
    // Inset as if inside the viewfinder (12dp margin + 12dp), so in 16:9, where it floats on the
    // picture, it keeps clear of the picture's edges and corners.
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
            QuickMoreToggle(quickOpen) {
                haptics.performHapticFeedback(if (quickOpen) HapticFeedbackType.ToggleOff else HapticFeedbackType.ToggleOn)
                onQuick(!quickOpen)
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
        // Concentric with the 56dp pill around it (4dp inset): pill at rest and checked alike.
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

/** Opens [QuickSettingsPanel]: a chevron that turns over while it is open, pointing down at the panel in any rotation. */
@Composable
private fun QuickMoreToggle(open: Boolean, onClick: () -> Unit) {
    val turn by animateFloatAsState(if (open) 180f else 0f, YoinMotion.defaultSpatialSpec(), label = "quickChevron")
    val label = stringResource(R.string.quick_settings)
    ToggleButton(
        checked = open,
        onCheckedChange = { onClick() },
        shapes = ToggleButtonDefaults.shapes(shape = PillShape, pressedShape = QuickTogglePressed, checkedShape = PillShape),
        colors = ToggleButtonDefaults.toggleButtonColors(containerColor = Color.Transparent),
        contentPadding = PaddingValues(0.dp),
        modifier = Modifier.size(width = 40.dp, height = 48.dp).semantics { contentDescription = label },
    ) {
        val tint = if (open) MaterialTheme.colorScheme.onSecondary else MaterialTheme.colorScheme.onSurface
        Canvas(Modifier.size(18.dp).graphicsLayer { rotationZ = turn }) {
            val w = size.width
            val h = size.height
            val path = Path().apply {
                moveTo(w * 0.2f, h * 0.38f)
                lineTo(w * 0.5f, h * 0.66f)
                lineTo(w * 0.8f, h * 0.38f)
            }
            drawPath(path, tint, style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
        }
    }
}

/**
 * The capture settings worth changing between shots, covering the viewfinder: output, processing,
 * burst frames, scene, look and Ultra HDR. With DNG only, nothing after output applies.
 */
@Composable
fun QuickSettingsPanel(
    settings: CameraSettings,
    onOutput: (OutputFormat) -> Unit,
    onProcessing: (Processing) -> Unit,
    onFrames: (BurstFrames) -> Unit,
    onScene: (SceneMode) -> Unit,
    onLook: (LookStyle) -> Unit,
    onUltraHdr: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val merged = settings.output != OutputFormat.DNG
    Column(
        modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            // Taps on the panel stay on it rather than reaching the viewfinder under it.
            .clickable(interactionSource = null, indication = null) {}
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
    ) {
        QuickRow(stringResource(R.string.settings_output)) {
            ConnectedChoice(OutputFormat.entries.filter { it.available }, settings.output, { outputLabel(it) }, onOutput, weight = { if (it == OutputFormat.DNG_JPEG) 1.6f else 1f })
        }
        QuickRow(stringResource(R.string.settings_scene), merged) {
            ConnectedChoice(SceneMode.entries, settings.scene, {
                stringResource(
                    when (it) {
                        SceneMode.AUTO -> R.string.scene_auto
                        SceneMode.BRACKET -> R.string.scene_bracket_short
                        SceneMode.NIGHT -> R.string.scene_night_short
                    },
                )
            }, onScene, enabled = merged)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            QuickRow(stringResource(R.string.settings_pipeline), merged, Modifier.weight(1f)) {
                ConnectedChoice(Processing.entries, settings.processing, { if (it == Processing.HDR_PLUS) "HDR+" else "HDR+ KR" }, onProcessing, enabled = merged)
            }
            QuickRow(stringResource(R.string.settings_burst_frames), merged, Modifier.weight(1f)) {
                ConnectedChoice(BurstFrames.entries, settings.burstFrames, { it.count.toString() }, onFrames, enabled = merged)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // night always finishes with its own look
            val lookApplies = merged && settings.scene != SceneMode.NIGHT
            QuickRow(stringResource(R.string.settings_look), lookApplies, Modifier.weight(1f)) {
                ConnectedChoice(LookStyle.entries, settings.look, {
                    if (it == LookStyle.PIXEL) "Pixel" else stringResource(R.string.look_standard)
                }, onLook, enabled = lookApplies)
            }
            QuickRow(stringResource(R.string.settings_ultra_hdr), merged, Modifier.weight(1f)) {
                ConnectedChoice(listOf(false, true), settings.ultraHdr, {
                    stringResource(if (it) R.string.value_on else R.string.value_off)
                }, onUltraHdr, enabled = merged)
            }
        }
    }
}

@Composable
private fun QuickRow(caption: String, enabled: Boolean = true, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            caption,
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = if (enabled) 1f else 0.38f),
            modifier = Modifier.padding(start = 4.dp),
        )
        content()
    }
}

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

/**
 * Auto reads "Auto" when picked, then, once AE has had [FLASH_METER_MS] to meter the scene, what
 * it decided: "A/On" or "A/Off", the A in bold.
 */
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
    // A tap shows the current lens's 35mm-equivalent focal length for a while, then its zoom factor again.
    var focalTap by remember { mutableIntStateOf(0) }
    var showFocal by remember { mutableStateOf(false) }
    LaunchedEffect(focalTap) {
        if (focalTap == 0) return@LaunchedEffect
        showFocal = true
        delay(FOCAL_SHOW_MS)
        showFocal = false
    }
    // Centred on its width with zoom factors. A lens showing its focal length grows outwards from
    // where it sits (the leftmost leftwards, the rightmost rightwards), the pill behind with it.
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
                        // Only the swap between focal lengths and zoom factors animates, not a zoom.
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

/** What a shutter press will write and through which pipeline, with the live exposure. */
@Composable
fun DevChip(settings: CameraSettings, camera: CameraInfo?, live: LiveExposure, modifier: Modifier = Modifier) {
    if (camera == null) return
    val numbers = MaterialTheme.typography.labelMedium.withTabularFigures()
    val quiet = MaterialTheme.colorScheme.onSurfaceVariant
    DevPill(modifier) {
        Text(outputLabel(settings.output), style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary, softWrap = false)
        Text(pipelineLabel(settings, camera, live), style = numbers, softWrap = false)
        live.iso?.let { Text("ISO $it", style = numbers, color = quiet, softWrap = false) }
        live.exposureNs?.let { Text(CameraFormat.shutter(it), style = numbers, color = quiet, softWrap = false) }
    }
}

private fun outputLabel(output: OutputFormat): String = when (output) {
    OutputFormat.DNG -> "DNG"
    OutputFormat.JPEG -> "JPEG"
    OutputFormat.DNG_JPEG -> "DNG+JPEG"
}

/** The pipeline [CameraViewModel] will pick for the next press, flash included. */
@Composable
private fun pipelineLabel(settings: CameraSettings, camera: CameraInfo, live: LiveExposure): String {
    if (settings.output == OutputFormat.DNG) return stringResource(R.string.pipeline_single)
    val merge = when (settings.processing) {
        Processing.HDR_PLUS -> "HDR+"
        Processing.HDR_PLUS_KR -> "HDR+ KR"
    }
    val flash = CameraViewModel.flashFires(settings, camera, live)
    val scene = when {
        flash -> stringResource(R.string.flash_short)
        settings.scene == SceneMode.BRACKET -> stringResource(R.string.scene_bracket_short)
        settings.scene == SceneMode.NIGHT -> stringResource(R.string.scene_night_short)
        else -> null
    }
    return listOfNotNull("$merge ×${if (flash) 1 else settings.burstFrames.count}", scene).joinToString(" ")
}

/** RAW requests left on this open of the device; a tap reopens it to give them all back. */
@Composable
fun BudgetPill(remaining: Int, limit: Int, enabled: Boolean, onRestart: () -> Unit, modifier: Modifier = Modifier) {
    val haptics = LocalHapticFeedback.current
    val color by animateColorAsState(
        if (remaining <= 5) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
        YoinMotion.defaultEffectsSpec(),
        label = "budgetColor",
    )
    var restarts by remember { mutableIntStateOf(0) }
    val spin by animateFloatAsState(restarts * 360f, YoinMotion.defaultSpatialSpec(), label = "budgetSpin")
    DevPill(
        modifier,
        spacing = 4.dp,
        enabled = enabled,
        onClick = {
            haptics.performHapticFeedback(HapticFeedbackType.Confirm)
            restarts++
            onRestart()
        },
    ) {
        Icon(YoinSymbols.Refresh, contentDescription = null, tint = color, modifier = Modifier.size(14.dp).graphicsLayer { rotationZ = spin })
        Text("$remaining/$limit", style = MaterialTheme.typography.labelMedium.withTabularFigures(), color = color, softWrap = false)
    }
}

/** Frames and seconds of the last pipeline run. */
@Composable
fun RunPill(run: PipelineRun, modifier: Modifier = Modifier) {
    DevPill(modifier) {
        Text(
            stringResource(R.string.dev_processed, run.frames, run.millis / 1000f),
            style = MaterialTheme.typography.labelMedium.withTabularFigures(),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            softWrap = false,
        )
    }
}

@Composable
private fun DevPill(
    modifier: Modifier = Modifier,
    spacing: Dp = 10.dp,
    onClick: (() -> Unit)? = null,
    enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    Row(
        modifier
            .clip(PillShape)
            .background(controlContainer())
            .then(if (onClick != null) Modifier.clickable(enabled = enabled, onClick = onClick) else Modifier)
            .padding(horizontal = 12.dp, vertical = 5.dp),
        horizontalArrangement = Arrangement.spacedBy(spacing),
        verticalAlignment = Alignment.CenterVertically,
        content = content,
    )
}

/** AF state (primary once focused, quiet while it scans) and the lens's focus distance. */
@Composable
fun AfChip(live: LiveExposure, modifier: Modifier = Modifier, onClick: () -> Unit = {}) {
    val dioptres = live.focusDioptres ?: return
    val color by animateColorAsState(
        when {
            live.focusLocked -> MaterialTheme.colorScheme.primary
            live.focusing -> MaterialTheme.colorScheme.onSurface
            else -> MaterialTheme.colorScheme.onSurfaceVariant
        },
        YoinMotion.defaultEffectsSpec(),
        label = "afColor",
    )
    Row(
        modifier
            .clip(PillShape)
            .background(controlContainer())
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text("AF", style = MaterialTheme.typography.labelMedium, color = color, fontWeight = FontWeight.Bold)
        Text(CameraFormat.focusDistance(dioptres), style = MaterialTheme.typography.labelMedium.withTabularFigures())
    }
}

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

/** The modes, centred, with room either side for [start] and [end]. */
@Composable
fun ModeBar(
    mode: CaptureMode,
    onMode: (CaptureMode) -> Unit,
    modifier: Modifier = Modifier,
    start: @Composable () -> Unit = {},
    end: @Composable () -> Unit = {},
) {
    Row(modifier.fillMaxWidth().padding(horizontal = SIDE_PADDING), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f), contentAlignment = Alignment.CenterStart) { start() }
        Modes(mode, onMode)
        Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) { end() }
    }
}

private val SIDE_PADDING = 20.dp

/** A slider's label-and-value button: its padding either side, and between label and value. */
private val SLIDER_LABEL_GAP = 12.dp

@Composable
private fun Modes(mode: CaptureMode, onMode: (CaptureMode) -> Unit) {
    val haptics = LocalHapticFeedback.current
    Row {
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
                    .padding(horizontal = 20.dp, vertical = 5.dp),
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

/**
 * ISO and shutter on the camera's own third stops, each with an Auto toggle on its label. The shutter
 * runs slow to fast like a dial, logarithmic in time, with the handheld range given the most travel.
 * [metered] is what auto would use.
 */
@Composable
fun ProPanel(
    camera: CameraInfo?,
    iso: Int,
    isoAuto: Boolean,
    exposureNs: Long,
    shutterAuto: Boolean,
    metered: ManualExposure?,
    clipped: Boolean,
    onIso: (Int?) -> Unit,
    onShutter: (Long?) -> Unit,
    modifier: Modifier = Modifier,
) {
    val isoStops = CameraViewModel.isoStops(camera)
    val shutterStops = remember(camera) { CameraViewModel.shutterStops(camera).asReversed() }
    val shutterWeights = remember(shutterStops) { shutterStops.map(::shutterStopWeight) }
    val isoFull = remember(isoStops) { isoStops.map { isFullStop(log2(it / 100.0)) } }
    val shutterFull = remember(shutterStops) { shutterStops.map { isFullStop(log2(it / 1e9)) } }
    // Both rows share the widest label and the widest value, so the bars line up and the value
    // sits evenly between its label and its bar.
    val shutterLabel = stringResource(R.string.control_shutter)
    val autoLabel = stringResource(R.string.value_auto)
    val measurer = rememberTextMeasurer()
    val labelStyle = MaterialTheme.typography.labelLarge
    val valueStyle = MaterialTheme.typography.titleSmall.withTabularFigures().copy(fontWeight = FontWeight.SemiBold)
    val autoStyle = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
    val density = LocalDensity.current
    val labelWidth = remember(shutterLabel, labelStyle) {
        with(density) { listOf("ISO", shutterLabel).maxOf { measurer.measure(it, labelStyle).size.width }.toDp() }
    }
    val valueWidth = remember(isoStops, shutterStops, autoLabel, valueStyle) {
        val values = isoStops.map(Int::toString) + shutterStops.map(CameraFormat::shutter)
        with(density) {
            maxOf(values.maxOf { measurer.measure(it, valueStyle).size.width }, measurer.measure(autoLabel, autoStyle).size.width).toDp()
        }
    }
    Column(
        modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
            .clip(YoinContainerShapes.Card)
            .background(controlContainer())
            // The label and value pad themselves, as one button.
            .padding(end = 16.dp, top = 2.dp, bottom = 2.dp),
    ) {
        StopSlider(
            label = "ISO",
            value = if (isoAuto) metered?.iso?.toString() ?: "–" else iso.toString(),
            auto = isoAuto,
            clipped = clipped && isoAuto,
            index = isoStops.indexOf(CameraViewModel.snapIso(if (isoAuto) metered?.iso ?: iso else iso, camera)),
            count = isoStops.size,
            onIndex = { onIso(it?.let(isoStops::get)) },
            fullStops = isoFull,
            labelWidth = labelWidth,
            valueWidth = valueWidth,
        )
        StopSlider(
            label = shutterLabel,
            value = if (shutterAuto) metered?.exposureNs?.let(CameraFormat::shutter) ?: "–" else CameraFormat.shutter(exposureNs),
            auto = shutterAuto,
            clipped = clipped && shutterAuto,
            index = shutterStops.indexOf(CameraViewModel.snapShutter(if (shutterAuto) metered?.exposureNs ?: exposureNs else exposureNs, camera)),
            count = shutterStops.size,
            fullStops = shutterFull,
            weights = shutterWeights,
            onIndex = { onShutter(it?.let(shutterStops::get)) },
            labelWidth = labelWidth,
            valueWidth = valueWidth,
        )
    }
}

/** Full travel within three stops of 1/125 s (1/1000 – 1/15), narrowing to 0.4 at the extremes. */
private fun shutterStopWeight(ns: Long): Float {
    val stops = abs(log2(ns / 8_000_000.0)).toFloat()
    return (1f - 0.2f * (stops - 3f)).coerceIn(0.4f, 1f)
}

private fun isFullStop(stopsFromReference: Double): Boolean =
    abs(stopsFromReference - Math.round(stopsFromReference)) < 0.1

/**
 * A scale of [count] stops that is swiped, not dragged: the value moves by the swipe from where it
 * is, so touching the bar never jumps. On Auto the thumb shows [index] (the metered stop) greyed
 * out, and the first swipe starts from it and leaves Auto. Tapping the label and value toggles Auto,
 * keeping the shown value when leaving it. [weights] (one per stop, uniform when null) sets how
 * much travel each stop gets; [fullStops] get a long tick, the rest a dot. [clipped]: auto ran out
 * of range and cannot make up the other one's exposure.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun StopSlider(
    label: String,
    value: String,
    auto: Boolean,
    clipped: Boolean,
    index: Int,
    count: Int,
    onIndex: (Int?) -> Unit,
    fullStops: List<Boolean>,
    labelWidth: Dp,
    valueWidth: Dp,
    weights: List<Float>? = null,
) {
    val haptics = LocalHapticFeedback.current
    val marks = remember(count, weights) {
        val w = weights ?: List(count) { 1f }
        FloatArray(count).also { m -> for (i in 1 until count) m[i] = m[i - 1] + (w[i - 1] + w[i]) / 2 }
    }
    val range = (marks.lastOrNull() ?: 0f).coerceAtLeast(1f)
    val position = index.coerceIn(0, (count - 1).coerceAtLeast(0))
    val current by rememberUpdatedState(position)
    val emit by rememberUpdatedState(onIndex)
    val autoColor by animateColorAsState(
        when {
            clipped && auto -> MaterialTheme.colorScheme.error
            auto -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f)
        },
        YoinMotion.defaultEffectsSpec(),
        label = "autoColor",
    )
    val valueColor by animateColorAsState(
        when {
            clipped -> MaterialTheme.colorScheme.error
            auto -> MaterialTheme.colorScheme.onSurfaceVariant
            else -> MaterialTheme.colorScheme.onSurface
        },
        YoinMotion.defaultEffectsSpec(),
        label = "valueColor",
    )
    val sliderColors = if (auto) {
        val grey = MaterialTheme.colorScheme.onSurfaceVariant
        SliderDefaults.colors(thumbColor = grey, activeTrackColor = grey.copy(alpha = 0.38f))
    } else {
        SliderDefaults.colors()
    }
    val tickActive = if (auto) MaterialTheme.colorScheme.surface.copy(alpha = 0.6f) else MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.7f)
    val tickInactive = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
    Row(Modifier.height(SLIDER_ROW_HEIGHT), verticalAlignment = Alignment.CenterVertically) {
        Row(
            Modifier
                .fillMaxHeight()
                .clip(RoundedCornerShape(12.dp))
                .clickable(enabled = count > 0) {
                    haptics.performHapticFeedback(if (auto) HapticFeedbackType.ToggleOff else HapticFeedbackType.ToggleOn)
                    onIndex(if (auto) position else null)
                }
                // The value column is as wide as the widest value and the bar has its own inset, so
                // the end is tighter for the gap to match the start's.
                .padding(start = SLIDER_LABEL_GAP, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(SLIDER_LABEL_GAP),
        ) {
            Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, softWrap = false, modifier = Modifier.width(labelWidth))
            Column(Modifier.width(valueWidth)) {
                Text(
                    stringResource(R.string.value_auto),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = autoColor,
                    softWrap = false,
                )
                Text(
                    value,
                    style = MaterialTheme.typography.titleSmall.withTabularFigures(),
                    fontWeight = FontWeight.SemiBold,
                    color = valueColor,
                    softWrap = false,
                )
            }
        }
        Box(Modifier.weight(1f)) {
            Slider(
                value = marks.getOrElse(position) { 0f },
                onValueChange = {},
                valueRange = 0f..range,
                enabled = count > 0,
                colors = sliderColors,
                track = { state ->
                    Box {
                        SliderDefaults.Track(sliderState = state, colors = sliderColors)
                        Canvas(Modifier.matchParentSize()) {
                            val thumbX = state.value / range * size.width
                            val gap = 8.dp.toPx()
                            val minDotSpacing = 5.dp.toPx()
                            val endIndicator = size.width - 10.dp.toPx()
                            val cy = size.height / 2
                            for (i in 1 until count) {
                                val x = marks[i] / range * size.width
                                if (abs(x - thumbX) < gap || x > endIndicator) continue
                                val full = fullStops[i]
                                val spacing = (if (i < count - 1) marks[i + 1] - marks[i] else marks[i] - marks[i - 1]) / range * size.width
                                if (!full && spacing < minDotSpacing) continue
                                val color = if (x < thumbX) tickActive else tickInactive
                                if (full) {
                                    val h = size.height * 0.28f
                                    drawLine(color, Offset(x, cy - h), Offset(x, cy + h), 1.5.dp.toPx(), StrokeCap.Round)
                                } else {
                                    drawCircle(color, 1.dp.toPx(), Offset(x, cy))
                                }
                            }
                        }
                    }
                },
            )
            Box(
                Modifier
                    .matchParentSize()
                    .pointerInput(count, marks) {
                        if (count < 2) return@pointerInput
                        var at = 0f
                        var last = 0
                        detectHorizontalDragGestures(
                            onDragStart = { at = marks[current]; last = current },
                            onHorizontalDrag = { change, dx ->
                                change.consume()
                                at = (at + dx / size.width * range).coerceIn(0f, range)
                                val p = marks.indices.minBy { abs(marks[it] - at) }
                                if (p != last) {
                                    last = p
                                    haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
                                    emit(p)
                                }
                            },
                        )
                    },
            )
        }
    }
}

/** Tight enough for the panel to fit between the 4:3 picture and the wave; the thumb overhangs it. */
private val SLIDER_ROW_HEIGHT = 40.dp

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