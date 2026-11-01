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
import com.izum1k.pocopaw.ui.component.PawPage
import com.izum1k.pocopaw.ui.component.Section
import com.izum1k.pocopaw.ui.component.SettingRow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

@Composable
fun CapabilitiesScreen() {
    val context = LocalContext.current
    var cameras by remember { mutableStateOf<List<CameraInfo>>(emptyList()) }
    LaunchedEffect(Unit) {
        cameras = withContext(Dispatchers.IO) { context.container.catalog.cameras }
    }
    PawPage(title = stringResource(R.string.capabilities_title)) {
        items(cameras, key = { it.id }) { cam -> CameraSection(cam) }
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
