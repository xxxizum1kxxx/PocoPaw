package com.izum1k.pocopaw.feature.camera

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.LocalActivity
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
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
import androidx.compose.ui.unit.Constraints
import com.izum1k.pocopaw.ui.theme.YoinContainerShapes
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
import kotlin.math.max
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
    val evLabel = if (ui.mode == CaptureMode.PHOTO && ui.evIndex != 0 && current != null) CameraFormat.ev(ui.evIndex, current.evStep) else null
    val evDial = if (ui.mode == CaptureMode.PHOTO && current != null) {
        SwipeDial(ui.evIndex, current.evRange.lower, current.evRange.upper, "EV " + CameraFormat.ev(ui.evIndex, current.evStep))
    } else {
        null
    }
    // Chips keep their last value while they fade out at 0.
    var evChip by remember { mutableStateOf("") }
    if (evLabel != null) evChip = evLabel
    var wbChip by remember { mutableStateOf("") }
    if (ui.wbShift != 0) wbChip = CameraFormat.wb(ui.wbShift)
    var quickOpen by rememberSaveable { mutableStateOf(false) }
    val wbDial = SwipeDial(ui.wbShift, -CameraViewModel.WB_MAX_STEPS, CameraViewModel.WB_MAX_STEPS, "WB " + CameraFormat.wb(ui.wbShift))

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        Viewfinder(
            frame = frame,
            grid = settings.grid,
            gridStyle = settings.gridStyle,
            focus = ui.focus,
            focusLocked = live.focusLocked,
            focusHeld = ui.focusHeld,
            faces = remember(live.faces, current) {
                val cam = current ?: return@remember emptyList()
                live.faces.map { CameraViewModel.sensorToView(it, cam.sensorOrientation, cam.facing == LensFacing.FRONT) }
            },
            evLabel = evLabel,
            countdown = ui.countdown,
            shutterTick = ui.shutterTick,
            cameraState = cameraState,
            live = ui.busy != Busy.IDLE || cameraState.showing() == current?.id,
            processing = ui.busy == Busy.PROCESSING,
            processingProgress = { ui.progress },
            ev = evDial,
            wb = wbDial,
            controlRotation = rotation,
            onSurfaceAvailable = vm::onSurfaceAvailable,
            onSurfaceDestroyed = vm::onSurfaceDestroyed,
            onTap = vm::tapFocus,
            onZoom = vm::zoomBy,
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
                Column(
                    Modifier
                        .align(Alignment.TopStart)
                        .offset { IntOffset(0, ((TOP_BAR_HEIGHT - TOP_INSET + FLOAT_INSET).toPx() * frame.blowUp(tallest)).roundToInt()) },
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    PillReveal(settings.developerInfo && live.focusDioptres != null) { AfChip(live, rotation) }
                    PillReveal(evLabel != null && ui.focus == null) { EvChip(evChip, rotation) { vm.setEv(0) } }
                    PillReveal(ui.wbShift != 0) { EvChip(wbChip, rotation, tag = "WB") { vm.setWb(0) } }
                }
            },
            cover = {
                AnimatedVisibility(
                    quickOpen,
                    enter = fadeIn(YoinMotion.defaultEffectsSpec()) + scaleIn(YoinMotion.defaultSpatialSpec(), initialScale = 1.04f),
                    exit = fadeOut(YoinMotion.fastEffectsSpec()) + scaleOut(YoinMotion.defaultSpatialSpec(), targetScale = 1.04f),
                ) {
                    // From the top of the 4:3 picture, where it sits in every framing, as tall as its rows.
                    QuickSettingsPanel(
                        settings = settings,
                        onOutput = { vm.updateSettings { setOutput(it) } },
                        onProcessing = { vm.updateSettings { setProcessing(it) } },
                        onFrames = { vm.updateSettings { setBurstFrames(it) } },
                        onScene = { vm.updateSettings { setScene(it) } },
                        onLook = { vm.updateSettings { setLook(it) } },
                        onUltraHdr = { vm.updateSettings { setUltraHdr(it) } },
                        modifier = Modifier
                            .layout { measurable, constraints ->
                                val w = constraints.maxWidth
                                val h = (w * AspectRatio.FOUR_THREE.portraitRatio).roundToInt()
                                val placeable = measurable.measure(Constraints(minWidth = w, maxWidth = w, maxHeight = h))
                                layout(w, placeable.height) {
                                    val under = (TOP_BAR_HEIGHT + VIEWFINDER_GAP).toPx()
                                    placeable.place(0, (under * frame.blowUp(tallest)).roundToInt())
                                }
                            }
                            .clip(YoinContainerShapes.Panel),
                    )
                }
            },
        )

        BackHandler(quickOpen) { quickOpen = false }
        TopControls(
            settings = settings,
            hasFlash = current?.hasFlash == true,
            flashRequired = live.flashRequired,
            rotation = rotation,
            onFlash = { vm.setFlash(it) },
            onTimer = { vm.setTimer(it) },
            onAspect = { vm.setAspect(it) },
            onGrid = { vm.setGrid(it) },
            quickOpen = quickOpen,
            onQuick = { quickOpen = it },
            onSettings = {
                quickOpen = false
                nav.push(Route.Settings)
            },
            modifier = Modifier
                .statusBarsPadding()
                .offset { IntOffset(0, ((FLOAT_INSET - TOP_INSET).toPx() * frame.blowUp(tallest)).roundToInt()) },
        )

        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SnackbarHost(snackbar, Modifier.align(Alignment.CenterHorizontally))
            // In 4:3 the info and the lenses sit in the black under the picture. A taller framing
            // blowing up under them, or the Pro panel taking their place, floats them onto the
            // picture, [FLOAT_INSET] inside its bottom edge.
            var floatTop by remember { mutableFloatStateOf(0f) }
            val pro by animateFloatAsState(if (ui.mode == CaptureMode.PRO) 1f else 0f, YoinMotion.defaultSpatialSpec(), label = "proFloat")
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
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
                                placeable.place(0, -(lift * max(b, pro)).roundToInt())
                            }
                        },
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    if (settings.developerInfo) DevChip(settings, current, live)
                    Reveal(current?.facing == LensFacing.BACK) {
                        current?.let {
                            LensSelector(
                                lenses = ui.backCameras,
                                current = it,
                                zoom = ui.zoom,
                                rotation = rotation,
                                onSelect = vm::selectCamera,
                            )
                        }
                    }
                }
                Reveal(ui.mode == CaptureMode.PRO) {
                    ProPanel(
                        camera = current,
                        iso = ui.manualIso,
                        isoAuto = ui.isoAuto,
                        exposureNs = ui.manualExposureNs,
                        shutterAuto = ui.shutterAuto,
                        metered = CameraViewModel.priorityExposure(ui, live, current),
                        clipped = CameraViewModel.priorityClipped(ui, live, current),
                        onIso = vm::setIso,
                        onShutter = vm::setShutter,
                    )
                }
                ControlsWave(
                    capturing = ui.busy == Busy.CAPTURING,
                    modifier = Modifier.padding(horizontal = SIDE_MARGIN + 12.dp),
                )
                ModeBar(
                    mode = ui.mode,
                    onMode = vm::setMode,
                    start = {
                        PillReveal(settings.developerInfo && current != null) {
                            BudgetPill(
                                remaining = remaining,
                                limit = vm.rawLimit,
                                enabled = ui.busy == Busy.IDLE && cameraState is CameraController.State.Previewing,
                                onRestart = vm::restartCamera,
                            )
                        }
                    },
                    end = {
                        PillReveal(settings.developerInfo && ui.lastRun != null) {
                            ui.lastRun?.let { RunPill(it) }
                        }
                    },
                )
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

/** [TopControls]' height under the status bar: 56dp of controls under [TOP_INSET]. */
private val TOP_BAR_HEIGHT = 64.dp

/** Between the 4:3 picture and the controls above and below it. */
private val VIEWFINDER_GAP = 16.dp
private val SIDE_MARGIN = 12.dp

/** [TopControls]' own top padding, which is all that separates them from the picture's top in the tallest framing. */
private val TOP_INSET = 8.dp

/** How far inside the tallest framing's edges the controls floating on it keep. */
private val FLOAT_INSET = 18.dp
/** The camera whose frames are on screen; a reopen mid-capture (RAW budget) keeps the busy flag instead. */
private fun CameraController.State.showing(): String? = when (this) {
    is CameraController.State.Previewing -> cameraId
    is CameraController.State.Capturing -> cameraId
    else -> null
}

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
