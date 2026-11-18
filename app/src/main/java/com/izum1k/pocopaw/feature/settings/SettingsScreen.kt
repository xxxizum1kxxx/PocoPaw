package com.izum1k.pocopaw.feature.settings

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.BuildConfig
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.camera.AwbEstimatorMode
import com.izum1k.pocopaw.container
import com.izum1k.pocopaw.data.BurstFrames
import com.izum1k.pocopaw.data.ColorStyle
import com.izum1k.pocopaw.data.DngContent
import com.izum1k.pocopaw.data.GridStyle
import com.izum1k.pocopaw.data.KrScaleMode
import com.izum1k.pocopaw.data.LookStyle
import com.izum1k.pocopaw.data.OutputFormat
import com.izum1k.pocopaw.data.Processing
import com.izum1k.pocopaw.data.SceneMode
import com.izum1k.pocopaw.pipeline.AccelBackend
import com.izum1k.pocopaw.pipeline.DemosaicMethod
import com.izum1k.pocopaw.pipeline.FlowMethod
import com.izum1k.pocopaw.pipeline.TonemapMode
import com.izum1k.pocopaw.ui.component.ChoiceRow
import com.izum1k.pocopaw.ui.component.PawPage
import com.izum1k.pocopaw.ui.component.Reveal
import com.izum1k.pocopaw.ui.component.Section
import com.izum1k.pocopaw.ui.component.SettingRow
import com.izum1k.pocopaw.ui.component.SwitchRow
import com.izum1k.pocopaw.ui.navigation.LocalNavigator
import com.izum1k.pocopaw.ui.navigation.Route
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val repo = context.container.settings
    val settings by repo.settings.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val nav = LocalNavigator.current

    PawPage(title = stringResource(R.string.settings_title)) {
        item {
            Section(title = stringResource(R.string.settings_defaults)) {
                Text(
                    stringResource(R.string.settings_defaults_note),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_output),
                    options = OutputFormat.entries,
                    selected = settings.output,
                    label = { stringResource(outputLabel(it)) },
                    onSelect = { scope.launch { repo.setOutput(it) } },
                    optionEnabled = { it.available },
                    icon = YoinSymbols.Image,
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_dng_content),
                    options = DngContent.entries,
                    selected = settings.effectiveDngContent(),
                    label = { stringResource(dngContentLabel(it)) },
                    onSelect = { scope.launch { repo.setDngContent(it) } },
                    enabled = settings.output == OutputFormat.DNG_JPEG && settings.processing == Processing.HDR_PLUS,
                )
                Reveal(settings.output == OutputFormat.DNG_JPEG && settings.processing == Processing.HDR_PLUS_KR) {
                    Text(
                        stringResource(R.string.settings_dng_kr_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                    )
                }
                ChoiceRow(
                    title = stringResource(R.string.settings_scene),
                    options = SceneMode.entries,
                    selected = settings.scene,
                    label = { stringResource(sceneLabel(it)) },
                    onSelect = { scope.launch { repo.setScene(it) } },
                    enabled = settings.isBurstMode,
                )
                SwitchRow(
                    title = stringResource(R.string.settings_zsl),
                    value = stringResource(R.string.settings_zsl_value),
                    checked = settings.zslEnabled,
                    onCheckedChange = { scope.launch { repo.setZslEnabled(it) } },
                    enabled = settings.isBurstMode,
                )
                SwitchRow(
                    title = stringResource(R.string.settings_raw10),
                    value = stringResource(R.string.settings_raw10_value),
                    checked = settings.raw10Enabled,
                    onCheckedChange = { scope.launch { repo.setRaw10Enabled(it) } },
                )
            }
        }
        item {
            val merged = settings.isBurstMode
            Section(title = stringResource(R.string.settings_processing)) {
                ChoiceRow(
                    title = stringResource(R.string.settings_pipeline),
                    options = Processing.entries,
                    selected = settings.processing,
                    label = { stringResource(processingLabel(it)) },
                    onSelect = { scope.launch { repo.setProcessing(it) } },
                    icon = YoinSymbols.Camera,
                    enabled = merged,
                )
                Reveal(merged && settings.processing == Processing.HDR_PLUS_KR) {
                    ChoiceRow(
                        title = stringResource(R.string.settings_kr_scale),
                        options = KrScaleMode.entries,
                        selected = settings.krScale,
                        label = { stringResource(krScaleLabel(it)) },
                        onSelect = { scope.launch { repo.setKrScale(it) } },
                        enabled = settings.isKrScaleEnabled(),
                    )
                }
                ChoiceRow(
                    title = stringResource(R.string.settings_demosaic),
                    options = DemosaicMethod.entries,
                    selected = settings.demosaicMethod,
                    label = { stringResource(demosaicLabel(it)) },
                    onSelect = { scope.launch { repo.setDemosaicMethod(it) } },
                    enabled = settings.isDemosaicEnabled(),
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_burst_frames),
                    options = BurstFrames.entries,
                    selected = settings.burstFrames,
                    label = { stringResource(R.string.burst_frames_value, it.count) },
                    onSelect = { scope.launch { repo.setBurstFrames(it) } },
                    enabled = merged,
                )
                SwitchRow(
                    title = stringResource(R.string.settings_early_stop),
                    value = stringResource(R.string.settings_early_stop_value),
                    checked = settings.isEarlyStopEnabled() && settings.earlyStopEnabled,
                    onCheckedChange = { scope.launch { repo.setEarlyStopEnabled(it) } },
                    enabled = settings.isEarlyStopEnabled(),
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_flow),
                    options = FlowMethod.entries,
                    selected = settings.flowMethod,
                    label = { stringResource(flowLabel(it)) },
                    onSelect = { scope.launch { repo.setFlowMethod(it) } },
                    enabled = merged,
                )
                SwitchRow(
                    title = stringResource(R.string.settings_pmrid),
                    value = stringResource(R.string.settings_pmrid_value),
                    checked = settings.pmridEnabled,
                    onCheckedChange = { scope.launch { repo.setPmridEnabled(it) } },
                    enabled = settings.isPmridEnabled(),
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_awb),
                    options = AwbEstimatorMode.entries,
                    selected = settings.awbMode,
                    label = { stringResource(awbLabel(it)) },
                    onSelect = { scope.launch { repo.setAwbMode(it) } },
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_tonemap),
                    options = TonemapMode.entries,
                    selected = settings.tonemapMode,
                    label = { stringResource(tonemapLabel(it)) },
                    onSelect = { scope.launch { repo.setTonemapMode(it) } },
                    enabled = merged,
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_look),
                    options = LookStyle.entries,
                    selected = settings.look,
                    label = { stringResource(lookLabel(it)) },
                    onSelect = { scope.launch { repo.setLook(it) } },
                    icon = YoinSymbols.Palette,
                    // night always finishes with its own look
                    enabled = settings.isLookEnabled(),
                )
                SwitchRow(
                    title = stringResource(R.string.settings_semantic_seg),
                    value = stringResource(R.string.settings_semantic_seg_value),
                    checked = settings.semanticSegEnabled,
                    onCheckedChange = { scope.launch { repo.setSemanticSegEnabled(it) } },
                    enabled = merged,
                )
                SwitchRow(
                    title = stringResource(R.string.settings_ultra_hdr),
                    value = stringResource(R.string.settings_ultra_hdr_value),
                    checked = settings.ultraHdr,
                    onCheckedChange = { scope.launch { repo.setUltraHdr(it) } },
                    enabled = merged,
                )
            }
        }
        item {
            Section(title = stringResource(R.string.settings_viewfinder)) {
                ChoiceRow(
                    title = stringResource(R.string.settings_grid_style),
                    options = GridStyle.entries,
                    selected = settings.gridStyle,
                    label = { stringResource(gridLabel(it)) },
                    onSelect = { scope.launch { repo.setGridStyle(it) } },
                )
                SwitchRow(
                    title = stringResource(R.string.settings_developer_info),
                    value = stringResource(R.string.settings_developer_info_value),
                    checked = settings.developerInfo,
                    onCheckedChange = { scope.launch { repo.setDeveloperInfo(it) } },
                )
            }
        }
        item {
            Section(title = stringResource(R.string.settings_feedback)) {
                SwitchRow(
                    title = stringResource(R.string.settings_shutter_sound),
                    checked = settings.shutterSound,
                    onCheckedChange = { scope.launch { repo.setShutterSound(it) } },
                )
                SwitchRow(
                    title = stringResource(R.string.settings_haptics),
                    checked = settings.haptics,
                    onCheckedChange = { scope.launch { repo.setHaptics(it) } },
                )
            }
        }
        item {
            Section(title = stringResource(R.string.settings_appearance)) {
                ChoiceRow(
                    title = stringResource(R.string.settings_color_style),
                    options = ColorStyle.entries,
                    selected = settings.colorStyle,
                    label = { stringResource(if (it == ColorStyle.EXPRESSIVE) R.string.color_expressive else R.string.color_system) },
                    onSelect = { scope.launch { repo.setColorStyle(it) } },
                )
            }
        }
        item {
            val merged = settings.isBurstMode
            Section(title = stringResource(R.string.settings_developer)) {
                ChoiceRow(
                    title = stringResource(R.string.settings_accel_backend),
                    options = AccelBackend.entries,
                    selected = settings.accelBackend,
                    label = { stringResource(accelBackendLabel(it)) },
                    onSelect = { scope.launch { repo.setAccelBackend(it) } },
                    icon = YoinSymbols.Memory,
                    enabled = merged,
                )
                SwitchRow(
                    title = stringResource(R.string.settings_gpu),
                    value = stringResource(R.string.settings_gpu_value),
                    checked = settings.effectiveGpuAcceleration(),
                    onCheckedChange = { scope.launch { repo.setGpuAcceleration(it) } },
                    enabled = merged && (settings.accelBackend == AccelBackend.AUTO || settings.accelBackend == AccelBackend.NPU_DSP),
                )
                SettingRow(
                    title = stringResource(R.string.capabilities_title),
                    icon = YoinSymbols.Info,
                    onClick = { nav.push(Route.Capabilities) },
                )
                if (BuildConfig.DEBUG) {
                    SettingRow(
                        title = stringResource(R.string.p1_title),
                        value = stringResource(R.string.p1_entry_value),
                        icon = YoinSymbols.Camera,
                        onClick = { nav.push(Route.P1Captures) },
                    )
                }
            }
        }
        item {
            Section(title = stringResource(R.string.settings_about)) {
                SettingRow(title = stringResource(R.string.app_name), value = BuildConfig.VERSION_NAME)
                SettingRow(title = "Aperture", value = stringResource(R.string.credit_aperture))
                SettingRow(title = "Yoin", value = stringResource(R.string.credit_yoin))
                SettingRow(title = "Yoin Symbols", value = stringResource(R.string.credit_symbols))
                SettingRow(title = "Material Color Utilities", value = stringResource(R.string.credit_mcu))
                SettingRow(title = "Google Sans Flex", value = stringResource(R.string.credit_font))
            }
        }
    }
}

private fun outputLabel(f: OutputFormat): Int = when (f) {
    OutputFormat.DNG -> R.string.output_dng
    OutputFormat.JPEG -> R.string.output_jpeg
    OutputFormat.DNG_JPEG -> R.string.output_dng_jpeg
}

private fun dngContentLabel(c: DngContent): Int = when (c) {
    DngContent.MERGED -> R.string.dng_content_merged
    DngContent.REFERENCE -> R.string.dng_content_reference
}

private fun processingLabel(p: Processing): Int = when (p) {
    Processing.HDR_PLUS -> R.string.pipeline_hdrplus
    Processing.HDR_PLUS_KR -> R.string.pipeline_hdrplus_kr
}

private fun sceneLabel(s: SceneMode): Int = when (s) {
    SceneMode.AUTO -> R.string.scene_auto
    SceneMode.BRACKET -> R.string.scene_bracket
    SceneMode.NIGHT -> R.string.scene_night
}

private fun gridLabel(g: GridStyle): Int = when (g) {
    GridStyle.THIRDS -> R.string.grid_thirds
    GridStyle.GOLDEN -> R.string.grid_golden
    GridStyle.SQUARE -> R.string.grid_square
}

private fun krScaleLabel(m: KrScaleMode): Int = when (m) {
    KrScaleMode.ONE_X -> R.string.kr_scale_1x
    KrScaleMode.TWO_X -> R.string.kr_scale_2x
    KrScaleMode.AUTO -> R.string.kr_scale_auto
}

private fun demosaicLabel(d: DemosaicMethod): Int = when (d) {
    DemosaicMethod.RCD -> R.string.demosaic_rcd
    DemosaicMethod.AMAZE -> R.string.demosaic_amaze
    DemosaicMethod.MHC -> R.string.demosaic_mhc
}

private fun flowLabel(f: FlowMethod): Int = when (f) {
    FlowMethod.BLOCK_MATCH -> R.string.flow_block_match
    FlowMethod.SEA_RAFT -> R.string.flow_sea_raft
    FlowMethod.AUTO -> R.string.flow_auto
}

private fun awbLabel(a: AwbEstimatorMode): Int = when (a) {
    AwbEstimatorMode.HAL -> R.string.awb_hal
    AwbEstimatorMode.FFCC -> R.string.awb_ffcc
    AwbEstimatorMode.C5 -> R.string.awb_c5
}

private fun tonemapLabel(t: TonemapMode): Int = when (t) {
    TonemapMode.MERTENS -> R.string.tonemap_mertens
    TonemapMode.HDRNET -> R.string.tonemap_hdrnet
}

private fun lookLabel(l: LookStyle): Int = when (l) {
    LookStyle.STANDARD -> R.string.look_standard
    LookStyle.PIXEL -> R.string.look_pixel
    LookStyle.FILM_WARM -> R.string.look_film_warm
    LookStyle.CLASSIC_CHROME -> R.string.look_classic_chrome
    LookStyle.BW_SILVER -> R.string.look_bw_silver
}

private fun accelBackendLabel(b: AccelBackend): Int = when (b) {
    AccelBackend.AUTO -> R.string.accel_auto
    AccelBackend.CPU -> R.string.accel_cpu
    AccelBackend.GPU_OPENCL -> R.string.accel_gpu
    AccelBackend.NPU_DSP -> R.string.accel_npu
}
