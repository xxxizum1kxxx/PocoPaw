package com.izum1k.pocopaw.feature.camera

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialShapes
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.toShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.layout.layout
import androidx.compose.ui.layout.onPlaced
import androidx.compose.ui.layout.positionInRoot
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.LensFacing
import com.izum1k.pocopaw.container
import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.ui.component.PillReveal
import com.izum1k.pocopaw.ui.component.Reveal
import com.izum1k.pocopaw.ui.navigation.LocalNavigator
import com.izum1k.pocopaw.ui.navigation.Route
import com.izum1k.pocopaw.ui.theme.YoinMotion
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val appContext = context.applicationContext
    val vm: CameraViewModel = viewModel { CameraViewModel(appContext, context.container) }
    var granted by remember { mutableStateOf(hasCameraPermission(context)) }
    var asked by rememberSaveable { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        asked = true
        granted = ok
    }
    LifecycleResumeEffect(Unit) {
        granted = hasCameraPermission(context)
        onPauseOrDispose { }
    }
    LaunchedEffect(granted) {
        vm.onPermission(granted)
        if (!granted && !asked) launcher.launch(Manifest.permission.CAMERA)
    }
    if (granted) {
        CameraContent(vm)
    } else {
        PermissionGate(asked = asked, onRequest = { launcher.launch(Manifest.permission.CAMERA) })
    }
}

@Composable
private fun CameraContent(vm: CameraViewModel) {
    val context = LocalContext.current
    val nav = LocalNavigator.current
    val haptics = LocalHapticFeedback.current
    val ui by vm.ui.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val cameraState by vm.cameraState.collectAsStateWithLifecycle()
    val live by vm.live.collectAsStateWithLifecycle()
    val deviceRotation by rememberDeviceRotation()
    val rotation = animatedControlRotation(deviceRotation)
    val snackbar = remember { SnackbarHostState() }

    LifecycleResumeEffect(vm) {
        vm.onResume()
        onPauseOrDispose { vm.onPause() }
    }

    LaunchedEffect(live.focusLocked, ui.focus?.id) {
        if (ui.focus != null && live.focusLocked) haptics.performHapticFeedback(HapticFeedbackType.SegmentFrequentTick)
    }
    LaunchedEffect(ui.countdown) {
        if (ui.countdown != null) haptics.performHapticFeedback(HapticFeedbackType.SegmentTick)
    }
    val failedText = stringResource(R.string.message_capture_failed)
    val saveFailedText = stringResource(R.string.message_save_failed)
    LaunchedEffect(ui.messageId) {
        val m = ui.message ?: return@LaunchedEffect
        haptics.performHapticFeedback(HapticFeedbackType.Reject)
        snackbar.showSnackbar(if (m == CaptureMessage.CAPTURE_FAILED) failedText else saveFailedText)
        vm.consumeMessage()
    }
    LaunchedEffect(ui.galleryOpenId) {
        if (!ui.openGalleryPending) return@LaunchedEffect
        vm.consumeGalleryOpen()
        openGallery(context, ui.last, ui.recentUris)
    }
    val zoomAnim = remember { Animatable(1f) }
    val zoomSpec = YoinMotion.defaultSpatialSpec<Float>()
    LaunchedEffect(ui.zoomAnimId) {
        val target = ui.zoomTarget ?: return@LaunchedEffect
        zoomAnim.snapTo(ui.zoom)
        zoomAnim.animateTo(target, zoomSpec) { vm.setZoom(value) }
    }

    val current = ui.current
    val frame = rememberAspectFrame(settings.aspect)
    val tallest = remember { AspectRatio.entries.maxOf { it.portraitRatio } }
    val statusTop = WindowInsets.statusBars.asPaddingValues().calculateTopPadding()
    val evLabel = if (ui.evIndex != 0 && current != null) CameraFormat.ev(ui.evIndex, current.evStep) else null
    val evDial = if (current != null) {
        SwipeDial(ui.evIndex, current.evRange.lower, current.evRange.upper, "EV " + CameraFormat.ev(ui.evIndex, current.evStep))
    } else {
        null
    }
    var evChip by remember { mutableStateOf("") }
    if (evLabel != null) evChip = evLabel
    var wbChip by remember { mutableStateOf("") }
    if (ui.wbShift != 0) wbChip = CameraFormat.wb(ui.wbShift)
    var chipRotation by remember { mutableFloatStateOf(-deviceRotation.toFloat()) }
    var chipsShown by remember { mutableStateOf(true) }
    LaunchedEffect(deviceRotation) {
        if (chipRotation == -deviceRotation.toFloat()) return@LaunchedEffect
        chipsShown = false
        delay(CHIP_TURN_MS)
        chipRotation = -deviceRotation.toFloat()
        chipsShown = true
    }
    val wbDial = SwipeDial(ui.wbShift, -CameraViewModel.WB_MAX_STEPS, CameraViewModel.WB_MAX_STEPS, "WB " + CameraFormat.wb(ui.wbShift))

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        Viewfinder(
            frame = frame,
            grid = settings.grid,
            focus = ui.focus,
            focusLocked = live.focusLocked,
            focusHeld = ui.focusHeld,
            faces = remember(live.faces, current, ui.zoom) {
                val cam = current ?: return@remember emptyList()
                live.faces
                    .map { CameraViewModel.sensorToView(it, cam.sensorOrientation, cam.facing == LensFacing.FRONT, ui.zoom) }
                    .filter { it.right > 0f && it.left < 1f && it.bottom > 0f && it.top < 1f }
            },
            evLabel = evLabel,
            countdown = ui.countdown,
            shutterTick = ui.shutterTick,
            cameraState = cameraState,
            live = ui.busy != Busy.IDLE || cameraState.showing() == current?.id,
            processing = ui.backgroundProcessing || ui.busy == Busy.PROCESSING,
            processingBlocked = ui.busy == Busy.PROCESSING,
            processingProgress = { ui.progress },
            ev = evDial,
            wb = wbDial,
            controlRotation = rotation,
            onSurfaceAvailable = vm::onSurfaceAvailable,
            onSurfaceDestroyed = vm::onSurfaceDestroyed,
            onTap = vm::tapFocus,
            onEvSteps = { vm.setEv(ui.evIndex + it) },
            onWbSteps = { vm.setWb(ui.wbShift + it) },
            onRetry = vm::retry,
            modifier = Modifier
                .offset {
                    val under = (TOP_BAR_HEIGHT + VIEWFINDER_GAP).toPx()
                    IntOffset(0, (statusTop.toPx() + under * (1f - frame.blowUp(tallest))).roundToInt())
                }
                .padding(horizontal = SIDE_MARGIN),
            topOverlay = {
                val b = frame.blowUp(tallest)
                UprightBox(
                    chipRotation,
                    Modifier.padding(
                        top = (TOP_BAR_HEIGHT - TOP_INSET + FLOAT_INSET) * b,
                        bottom = CHIP_FLOAT_CLEARANCE * b,
                    ),
                ) {
                    Column(Modifier.align(Alignment.TopStart), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        PillReveal(chipsShown && evLabel != null && ui.focus == null) { EvChip(evChip) { vm.setEv(0) } }
                        PillReveal(chipsShown && ui.wbShift != 0) { EvChip(wbChip, tag = "WB") { vm.setWb(0) } }
                    }
                }
            },
        )

        TopControls(
            settings = settings,
            hasFlash = current?.hasFlash == true,
            flashRequired = live.flashRequired,
            rotation = rotation,
            onFlash = vm::setFlash,
            onTimer = vm::setTimer,
            onAspect = vm::setAspect,
            onGrid = vm::setGrid,
            onSettings = { nav.push(Route.Settings) },
            modifier = Modifier
                .statusBarsPadding()
                .offset { IntOffset(0, ((FLOAT_INSET - TOP_INSET).toPx() * frame.blowUp(tallest)).roundToInt()) },
        )

        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(bottom = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SnackbarHost(snackbar, Modifier.align(Alignment.CenterHorizontally))
            var floatTop by remember { mutableFloatStateOf(0f) }
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .onPlaced { floatTop = it.positionInRoot().y }
                        .layout { measurable, constraints ->
                            val placeable = measurable.measure(constraints)
                            layout(placeable.width, placeable.height) {
                                val b = frame.blowUp(tallest)
                                val top = statusTop.toPx() + (TOP_BAR_HEIGHT + VIEWFINDER_GAP).toPx() * (1f - b)
                                val bottom = top + (placeable.width - 2 * SIDE_MARGIN.toPx()) * frame.heightRatio
                                val lift = (floatTop + placeable.height - bottom + FLOAT_INSET.toPx()).coerceAtLeast(0f)
                                placeable.place(0, -(lift * b).roundToInt())
                            }
                        },
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Reveal(current?.facing == LensFacing.BACK) {
                        current?.let {
                            LensSelector(
                                lenses = ui.backCameras,
                                current = it,
                                zoom = ui.zoom,
                                rotation = rotation,
                                onSelect = vm::selectCamera,
                                onToggleZoom = vm::toggleTwoX,
                            )
                        }
                    }
                }
                ControlsWave(
                    capturing = ui.busy == Busy.CAPTURING,
                    modifier = Modifier.padding(horizontal = SIDE_MARGIN + 12.dp),
                )
                ShutterRow(
                    last = ui.last?.thumbnail,
                    hasLast = ui.last != null,
                    awaitingGallery = ui.awaitingGallery,
                    busy = ui.busy,
                    processing = ui.backgroundProcessing || ui.busy == Busy.PROCESSING,
                    progress = ui.progress,
                    countdown = ui.countdown,
                    canFlip = ui.canFlip,
                    rotation = rotation,
                    onThumbnail = vm::onThumbnailClick,
                    onShutter = { vm.shutter(deviceRotation) },
                    onFlip = vm::flip,
                )
            }
        }
    }
}

@Composable
private fun PermissionGate(asked: Boolean, onRequest: () -> Unit) {
    val context = LocalContext.current
    val activity = LocalActivity.current
    val haptics = LocalHapticFeedback.current
    val permanentlyDenied = asked && activity?.shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) == false
    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surface)
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp, Alignment.CenterVertically),
    ) {
        Box(
            Modifier
                .size(112.dp)
                .clip(MaterialShapes.Cookie9Sided.toShape())
                .background(MaterialTheme.colorScheme.primaryContainer),
            contentAlignment = Alignment.Center,
        ) {
            Icon(YoinSymbols.Camera, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimaryContainer, modifier = Modifier.size(48.dp))
        }
        Text(stringResource(R.string.permission_title), style = MaterialTheme.typography.headlineSmall)
        Button(
            onClick = {
                haptics.performHapticFeedback(HapticFeedbackType.Confirm)
                if (permanentlyDenied) {
                    context.startActivity(
                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                    )
                } else {
                    onRequest()
                }
            },
            shapes = ButtonDefaults.shapes(),
        ) {
            Text(stringResource(if (permanentlyDenied) R.string.permission_open_settings else R.string.permission_grant))
        }
    }
}

private val TOP_BAR_HEIGHT = 64.dp
private val VIEWFINDER_GAP = 16.dp
private val SIDE_MARGIN = 12.dp
private val TOP_INSET = 8.dp
private val FLOAT_INSET = 18.dp
private val CHIP_FLOAT_CLEARANCE = 72.dp
private const val CHIP_TURN_MS = 220L

private fun CameraController.State.showing(): String? = when (this) {
    is CameraController.State.Previewing -> cameraId
    is CameraController.State.Capturing -> cameraId
    else -> null
}

private fun hasCameraPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
