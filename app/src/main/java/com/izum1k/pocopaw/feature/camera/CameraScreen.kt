package com.izum1k.pocopaw.feature.camera

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.layout.boundsInRoot
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
import com.izum1k.pocopaw.ui.component.Reveal
import com.izum1k.pocopaw.ui.navigation.LocalNavigator
import com.izum1k.pocopaw.ui.navigation.Route
import kotlin.math.roundToInt

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val vm: CameraViewModel = viewModel { CameraViewModel(context.container) }
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
    val remaining by vm.rawRemaining.collectAsStateWithLifecycle()
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

    val current = ui.current
    val frame = rememberAspectFrame(settings.aspect)
    val tallest = remember { AspectRatio.entries.maxOf { it.portraitRatio } }
    val statusTop = WindowInsets.statusBars.asPaddingValues().calculateTopPadding()
    var viewfinderBottom by remember { mutableFloatStateOf(0f) }
    var controlsTop by remember { mutableFloatStateOf(0f) }
    val evLabel = if (ui.mode == CaptureMode.PHOTO && ui.evIndex != 0 && current != null) CameraFormat.ev(ui.evIndex, current.evStep) else null

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        Viewfinder(
            frame = frame,
            grid = settings.grid,
            gridStyle = settings.gridStyle,
            focus = ui.focus,
            focusLocked = live.focusLocked,
            evLabel = evLabel,
            countdown = ui.countdown,
            shutterTick = ui.shutterTick,
            cameraState = cameraState,
            evDragEnabled = ui.mode == CaptureMode.PHOTO,
            controlRotation = rotation,
            onSurfaceAvailable = vm::onSurfaceAvailable,
            onSurfaceDestroyed = vm::onSurfaceDestroyed,
            onTap = vm::tapFocus,
            onZoom = vm::zoomBy,
            onEvSteps = { vm.setEv(ui.evIndex + it) },
            onRetry = vm::retry,
            modifier = Modifier
                .offset {
                    val under = (TOP_BAR_HEIGHT + VIEWFINDER_GAP).toPx()
                    IntOffset(0, (statusTop.toPx() + under * (1f - frame.blowUp(tallest))).roundToInt())
                }
                .padding(horizontal = SIDE_MARGIN)
                .onPlaced { viewfinderBottom = it.boundsInRoot().bottom },
            topOverlay = {
                Column(
                    Modifier
                        .align(Alignment.TopStart)
                        .offset { IntOffset(0, ((TOP_BAR_HEIGHT + VIEWFINDER_GAP - 12.dp).toPx() * frame.blowUp(tallest)).roundToInt()) },
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (settings.developerInfo) DevChip(current, remaining, vm.rawLimit, live, ui.lastRun)
                    Reveal(evLabel != null && ui.focus == null) { EvChip(evLabel.orEmpty(), rotation) }
                }
            },
            bottomOverlay = {
                if (current?.facing == LensFacing.BACK) {
                    LensSelector(
                        lenses = ui.backCameras,
                        current = current,
                        zoom = ui.zoom,
                        rotation = rotation,
                        onSelect = vm::selectCamera,
                        modifier = Modifier.align(Alignment.BottomCenter),
                    )
                }
            },
        )

        TopControls(
            settings = settings,
            hasFlash = current?.hasFlash == true,
            rotation = rotation,
            onFlash = { vm.setFlash(it) },
            onTimer = { vm.setTimer(it) },
            onAspect = { vm.setAspect(it) },
            onGrid = { vm.setGrid(it) },
            onSettings = { nav.push(Route.Settings) },
            modifier = Modifier.statusBarsPadding(),
        )

        ControlsWave(
            live = cameraState is CameraController.State.Previewing || cameraState is CameraController.State.Capturing,
            busy = ui.busy == Busy.CAPTURING || ui.busy == Busy.PROCESSING,
            modifier = Modifier
                .padding(horizontal = SIDE_MARGIN + 12.dp)
                .layout { measurable, constraints ->
                    val placeable = measurable.measure(constraints)
                    layout(placeable.width, placeable.height) {
                        val gap = controlsTop - viewfinderBottom
                        val y = viewfinderBottom + (gap - placeable.height) / 2f
                        val shown = ((gap - placeable.height - WAVE_CLEARANCE.toPx()) / WAVE_CLEARANCE.toPx()).coerceIn(0f, 1f)
                        placeable.placeWithLayer(0, y.roundToInt()) { alpha = shown }
                    }
                },
        )

        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SnackbarHost(snackbar, Modifier.align(Alignment.CenterHorizontally))
            Column(
                Modifier.onPlaced { controlsTop = it.positionInRoot().y },
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Reveal(ui.mode == CaptureMode.PRO) {
                    ProPanel(
                        camera = current,
                        iso = ui.manualIso,
                        exposureNs = ui.manualExposureNs,
                        onIso = vm::setIso,
                        onShutter = vm::setShutter,
                    )
                }
                ModeBar(ui.mode, vm::setMode)
                ShutterRow(
                    last = ui.last?.thumbnail,
                    hasLast = ui.last != null,
                    busy = ui.busy,
                    countdown = ui.countdown,
                    canFlip = ui.canFlip,
                    rotation = rotation,
                    onThumbnail = { ui.last?.uri?.let { openCapture(context, it) } },
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

/** [TopControls]' height under the status bar: 56dp of controls and 8dp above and below. */
private val TOP_BAR_HEIGHT = 72.dp
private val VIEWFINDER_GAP = 4.dp
private val SIDE_MARGIN = 12.dp

/** The wave fades in over this much more room than it needs, and is gone with less. */
private val WAVE_CLEARANCE = 12.dp

private fun hasCameraPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

private fun openCapture(context: Context, uri: Uri) {
    val dng = Intent(Intent.ACTION_VIEW).setDataAndType(uri, "image/x-adobe-dng").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    try {
        context.startActivity(dng)
    } catch (_: ActivityNotFoundException) {
        runCatching {
            context.startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(uri, "image/*").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
        }
    }
}
