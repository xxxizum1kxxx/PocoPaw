package com.izum1k.pocopaw.feature.p1

import android.graphics.SurfaceTexture
import android.view.TextureView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.LoadingIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.container
import com.izum1k.pocopaw.feature.camera.Busy
import com.izum1k.pocopaw.feature.camera.CameraFormat
import com.izum1k.pocopaw.feature.camera.ShutterButton
import com.izum1k.pocopaw.ui.component.BackButton
import com.izum1k.pocopaw.ui.component.ConnectedChoice
import com.izum1k.pocopaw.ui.theme.YoinContainerShapes
import kotlinx.coroutines.delay
import java.util.Locale

@Composable
fun P1CaptureScreen(name: String) {
    val item = P1Items.byName(name) ?: return
    val context = LocalContext.current
    val vm: P1CaptureViewModel = viewModel { P1CaptureViewModel(context.container, P1Store(context), item) }
    val ui by vm.ui.collectAsStateWithLifecycle()
    val cameraState by vm.cameraState.collectAsStateWithLifecycle()

    LifecycleResumeEffect(vm) {
        vm.onResume()
        onPauseOrDispose { vm.onPause() }
    }
    LaunchedEffect(ui.message) {
        if (ui.message != null) {
            delay(MESSAGE_MS)
            vm.consumeMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(item.name, fontFamily = FontFamily.Monospace) },
                navigationIcon = { BackButton() },
                actions = {
                    Text(
                        lensLabel(item),
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(end = 16.dp),
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
            )
        },
        containerColor = MaterialTheme.colorScheme.surface,
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            Preview(item, vm, ui, cameraState)
            Spacer(Modifier.weight(1f))
            Column(
                Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, bottom = 12.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                if (item.variants.size > 1) {
                    ConnectedChoice(
                        options = item.variants.indices.toList(),
                        selected = ui.variant,
                        label = { variantLabel(item.variants[it]) },
                        onSelect = vm::selectVariant,
                        enabled = !ui.busy,
                    )
                }
                ShutterButton(busy = if (ui.busy) Busy.CAPTURING else Busy.IDLE, countdown = null, onClick = vm::shutter)
            }
        }
    }
}

@Composable
private fun Preview(item: P1Item, vm: P1CaptureViewModel, ui: P1CaptureUiState, cameraState: CameraController.State) {
    var tap by remember { mutableStateOf<Offset?>(null) }
    LaunchedEffect(tap) {
        if (tap != null) {
            delay(FOCUS_RING_MS)
            tap = null
        }
    }
    val busy by rememberUpdatedState(ui.busy)
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(3f / 4f)
            .clip(YoinContainerShapes.Panel)
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        PreviewTexture(vm::onSurfaceAvailable, vm::onSurfaceDestroyed, Modifier.fillMaxSize())
        Box(
            Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures { p ->
                        if (busy) return@detectTapGestures
                        tap = p
                        vm.tapFocus(p.x / size.width, p.y / size.height)
                    }
                },
        )
        tap?.let { p ->
            Canvas(Modifier.fillMaxSize()) {
                drawCircle(Color.White, radius = 36.dp.toPx(), center = p, style = Stroke(2.dp.toPx()))
            }
        }
        when (cameraState) {
            is CameraController.State.Opening -> LoadingIndicator()
            is CameraController.State.Error -> Button(onClick = vm::retry) { Text(stringResource(R.string.action_retry)) }
            else -> Unit
        }
        Requirements(item, ui, Modifier.align(Alignment.TopCenter))
        Status(ui, Modifier.align(Alignment.BottomCenter))
    }
}

/** The press's requirements, over the top of the preview. */
@Composable
private fun Requirements(item: P1Item, ui: P1CaptureUiState, modifier: Modifier) {
    val v = item.variants[ui.variant]
    val required = ui.variant == 0
    Column(
        modifier
            .fillMaxWidth()
            .background(Color.Black.copy(alpha = OVERLAY_ALPHA))
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            item.scene?.let { stringResource(item.group.title) + " · " + stringResource(it) } ?: stringResource(item.group.title),
            style = MaterialTheme.typography.titleSmall,
            color = Color.White,
        )
        Text(stringResource(item.group.note), style = MaterialTheme.typography.bodySmall, color = Color.White.copy(alpha = 0.75f))
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 2.dp)) {
            Text(
                stringResource(if (required) R.string.p1_required else R.string.p1_optional),
                style = MaterialTheme.typography.labelSmall,
                color = if (required) MaterialTheme.colorScheme.onPrimary else Color.White,
                modifier = Modifier
                    .clip(YoinContainerShapes.Card)
                    .background(if (required) MaterialTheme.colorScheme.primary else Color.White.copy(alpha = 0.2f))
                    .padding(horizontal = 6.dp, vertical = 2.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(
                if (item.variants.size > 1) variantLabel(v) + " · " + variantDescription(v) else variantDescription(v),
                style = MaterialTheme.typography.labelLarge,
                color = Color.White,
                modifier = Modifier.weight(1f),
            )
            Text(
                stringResource(R.string.p1_sets, ui.counts.getOrElse(ui.variant) { 0 }),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(start = 8.dp),
            )
        }
    }
}

/** Progress while capturing, else the last burst or what went wrong, over the bottom of the preview. */
@Composable
private fun Status(ui: P1CaptureUiState, modifier: Modifier) {
    val last = ui.last
    if (!ui.busy && ui.message == null && last == null) return
    Column(
        modifier
            .fillMaxWidth()
            .background(Color.Black.copy(alpha = OVERLAY_ALPHA))
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        when {
            ui.busy -> {
                Text(stringResource(R.string.p1_writing, ui.written, ui.total), color = Color.White, style = MaterialTheme.typography.labelLarge)
                LinearProgressIndicator(
                    progress = { if (ui.total > 0) ui.written / ui.total.toFloat() else 0f },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            ui.message != null -> Text(
                stringResource(if (ui.message == P1Message.METERING) R.string.p1_metering else R.string.p1_failed),
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.labelLarge,
            )
            last != null -> Text(
                stringResource(
                    R.string.p1_result,
                    last.folder,
                    last.iso ?: 0,
                    last.exposureNs?.let(CameraFormat::shutter) ?: "–",
                    String.format(Locale.US, "%.3f", last.gyroRms),
                ),
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = Color.White,
            )
        }
    }
}

@Composable
private fun PreviewTexture(onAvailable: (SurfaceTexture) -> Unit, onDestroyed: () -> Unit, modifier: Modifier) {
    val available by rememberUpdatedState(onAvailable)
    val destroyed by rememberUpdatedState(onDestroyed)
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

private const val MESSAGE_MS = 3_000L
private const val FOCUS_RING_MS = 1_000L
private const val OVERLAY_ALPHA = 0.55f
