package com.izum1k.pocopaw.feature.settings

import android.hardware.camera2.CameraMetadata
import androidx.compose.foundation.lazy.items
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.LensFacing
import com.izum1k.pocopaw.camera.RawRequestBudget
import com.izum1k.pocopaw.container
import com.izum1k.pocopaw.feature.camera.CameraFormat
import com.izum1k.pocopaw.pipeline.AccelCapabilities
import com.izum1k.pocopaw.pipeline.LutCatalog
import com.izum1k.pocopaw.pipeline.ModelCatalog
import com.izum1k.pocopaw.pipeline.NativeBurst
import com.izum1k.pocopaw.ui.component.PawPage
import com.izum1k.pocopaw.ui.component.Section
import com.izum1k.pocopaw.ui.component.SettingRow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

private data class BundledAssetStatus(
    val relPath: String,
    val bytes: Long?,
)

@Composable
fun CapabilitiesScreen() {
    val context = LocalContext.current
    var cameras by remember { mutableStateOf<List<CameraInfo>>(emptyList()) }
    var gpu by remember { mutableStateOf<String?>(null) }
    var dsp by remember { mutableStateOf<String?>(null) }
    var accel by remember { mutableStateOf<String?>(null) }
    var assets by remember { mutableStateOf<List<BundledAssetStatus>?>(null) }
    LaunchedEffect(Unit) {
        cameras = withContext(Dispatchers.IO) { context.container.catalog.cameras }
        gpu = context.container.gpuInfo.await()
        dsp = withContext(Dispatchers.IO) { NativeBurst.dspProbe() }
        accel = withContext(Dispatchers.IO) { NativeBurst.accelProbe() }
        assets = withContext(Dispatchers.IO) {
            val paths = listOf(
                "${ModelCatalog.ASSET_DIR}/${ModelCatalog.PMRID_FILE}",
                "${ModelCatalog.ASSET_DIR}/${ModelCatalog.SEA_RAFT_FILE}",
                "${ModelCatalog.ASSET_DIR}/${ModelCatalog.HDRNET_FILE}",
                "${ModelCatalog.ASSET_DIR}/${ModelCatalog.SEG_FILE}",
                "${ModelCatalog.ASSET_DIR}/c5_m7.onnx",
            ) + LutCatalog.presets.map { "${LutCatalog.ASSET_DIR}/${it.fileName}" } + listOf(
                "calib/ffcc_cam0.bin",
                "calib/ffcc_cam2.bin",
            )
            paths.map { BundledAssetStatus(it, context.container.checkAssetBytes(it)) }
        }
    }
    PawPage(title = stringResource(R.string.capabilities_title)) {
        items(cameras, key = { it.id }) { cam -> CameraSection(cam) }
        item(key = "gpu") { ProbeSection(stringResource(R.string.cap_gpu), gpu) }
        item(key = "dsp") { ProbeSection(stringResource(R.string.cap_dsp), dsp) }
        item(key = "accel") { AccelSection(accel) }
        item(key = "assets") { AssetsSection(assets) }
    }
}

@Composable
private fun AccelSection(report: String?) {
    Section(title = stringResource(R.string.cap_accel)) {
        if (report == null) {
            SettingRow(title = stringResource(R.string.cap_probing))
        } else {
            val cap = AccelCapabilities.parse(report)
            val unavailable = stringResource(R.string.cap_accel_unavailable)
            SettingRow(
                title = stringResource(R.string.cap_accel_opencl),
                value = cap.openclDevice.ifEmpty { unavailable },
            )
            SettingRow(
                title = stringResource(R.string.cap_accel_ort),
                value = if (cap.ortAvailable) cap.ortVersion else unavailable,
            )
            SettingRow(
                title = stringResource(R.string.cap_accel_providers),
                value = cap.ortProviders.joinToString(", ").ifEmpty { unavailable },
            )
            SettingRow(
                title = stringResource(R.string.cap_accel_nnapi),
                value = yesNo(cap.nnapiAvailable),
            )
            SettingRow(
                title = stringResource(R.string.cap_accel_qnn),
                value = cap.qnnNote.ifEmpty { yesNo(cap.qnnAvailable) },
            )
            SettingRow(
                title = stringResource(R.string.cap_accel_threads),
                value = if (cap.bigPrimeCpuIds.isNotEmpty()) {
                    stringResource(R.string.cap_accel_threads_cpus, cap.mlThreads, cap.bigPrimeCpuIds.joinToString(","))
                } else {
                    cap.mlThreads.toString()
                },
            )
            SettingRow(
                title = stringResource(R.string.cap_accel_cdsp),
                value = cap.cdspSummary.ifEmpty { yesNo(cap.cdspFastRpcAvailable) },
            )
            if (cap.hexagonSdkBoundary.isNotEmpty()) {
                SettingRow(
                    title = stringResource(R.string.cap_accel_boundary),
                    value = cap.hexagonSdkBoundary,
                )
            }
        }
    }
}

@Composable
private fun AssetsSection(items: List<BundledAssetStatus>?) {
    Section(title = stringResource(R.string.cap_assets)) {
        if (items == null) {
            SettingRow(title = stringResource(R.string.cap_probing))
        } else {
            for (asset in items) {
                val status = when (val b = asset.bytes) {
                    null -> stringResource(R.string.cap_asset_missing)
                    else -> if (b >= 1_000_000L) {
                        stringResource(R.string.cap_asset_ready_mb, b / (1024f * 1024f))
                    } else {
                        stringResource(R.string.cap_asset_ready_kb, (b / 1024L).coerceAtLeast(1L))
                    }
                }
                SettingRow(title = asset.relPath, value = status)
            }
        }
    }
}

/** "property: value" lines of a native probe; null while it runs. */
@Composable
private fun ProbeSection(title: String, report: String?) {
    Section(title = title) {
        if (report == null) {
            SettingRow(title = stringResource(R.string.cap_probing))
        } else {
            for (line in report.lines().filter { it.isNotBlank() }) {
                val colon = line.indexOf(": ")
                if (colon < 0) SettingRow(title = line) else SettingRow(title = line.substring(0, colon), value = line.substring(colon + 2))
            }
        }
    }
}

@Composable
private fun CameraSection(cam: CameraInfo) {
    val title = when (cam.facing) {
        LensFacing.FRONT -> stringResource(R.string.lens_front)
        LensFacing.BACK -> stringResource(R.string.lens_back, CameraFormat.zoom(cam.relativeZoom))
        LensFacing.EXTERNAL -> stringResource(R.string.lens_external)
    }
    Section(title = title) {
        SettingRow(title = stringResource(R.string.cap_id), value = cam.id)
        SettingRow(
            title = stringResource(R.string.cap_raw_size),
            value = String.format(Locale.US, "%d×%d  %.1f MP", cam.rawSize.width, cam.rawSize.height, cam.rawMegapixels),
        )
        SettingRow(title = stringResource(R.string.cap_cfa), value = cam.cfa.name)
        SettingRow(title = stringResource(R.string.cap_levels), value = "${cam.blackLevel} / ${cam.whiteLevel}")
        SettingRow(
            title = stringResource(R.string.cap_focal),
            value = CameraFormat.millimetres(cam.focalLengthMm) + "  " + stringResource(R.string.cap_equivalent, cam.equivalentFocalMm.toInt()),
        )
        cam.aperture?.let { SettingRow(title = stringResource(R.string.cap_aperture), value = String.format(Locale.US, "f/%.1f", it)) }
        cam.isoRange?.let { SettingRow(title = stringResource(R.string.cap_iso), value = "${it.lower} – ${it.upper}") }
        cam.exposureRangeNs?.let {
            SettingRow(title = stringResource(R.string.cap_exposure), value = "${CameraFormat.shutter(it.lower)} – ${CameraFormat.shutter(it.upper)}")
        }
        SettingRow(title = stringResource(R.string.cap_preview), value = "${cam.previewSize.width}×${cam.previewSize.height}")
        SettingRow(title = stringResource(R.string.cap_hardware_level), value = hardwareLevel(cam.hardwareLevel))
        SettingRow(title = stringResource(R.string.cap_ois), value = yesNo(cam.hasOis))
        SettingRow(title = stringResource(R.string.cap_flash), value = yesNo(cam.hasFlash))
        SettingRow(title = stringResource(R.string.cap_af), value = yesNo(cam.hasAutoFocus))
        SettingRow(title = stringResource(R.string.cap_noise_profile), value = yesNo(cam.hasNoiseProfile))
        SettingRow(title = stringResource(R.string.cap_raw10), value = yesNo(cam.supportsRaw10))
        SettingRow(title = stringResource(R.string.cap_raw_budget), value = RawRequestBudget.DEFAULT_LIMIT.toString())
    }
}

@Composable
private fun yesNo(v: Boolean): String = stringResource(if (v) R.string.value_yes else R.string.value_no)

private fun hardwareLevel(level: Int): String = when (level) {
    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
    CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
    else -> level.toString()
}
