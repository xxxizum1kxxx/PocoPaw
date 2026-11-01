package com.izum1k.pocopaw.feature.settings

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.BuildConfig
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.container
import com.izum1k.pocopaw.data.ColorStyle
import com.izum1k.pocopaw.data.GridStyle
import com.izum1k.pocopaw.data.OutputFormat
import com.izum1k.pocopaw.pipeline.DngPassthroughPipeline
import com.izum1k.pocopaw.pipeline.PipelineRegistry
import com.izum1k.pocopaw.ui.component.ChoiceRow
import com.izum1k.pocopaw.ui.component.PawPage
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
            Section(title = stringResource(R.string.settings_capture)) {
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
                    title = stringResource(R.string.settings_pipeline),
                    options = PipelineRegistry.ids,
                    selected = PipelineRegistry.defaultId,
                    label = { stringResource(pipelineLabel(it)) },
                    onSelect = { },
                    icon = YoinSymbols.Camera,
                )
                ChoiceRow(
                    title = stringResource(R.string.settings_grid_style),
                    options = GridStyle.entries,
                    selected = settings.gridStyle,
                    label = { stringResource(gridLabel(it)) },
                    onSelect = { scope.launch { repo.setGridStyle(it) } },
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
            Section(title = stringResource(R.string.settings_developer)) {
                SwitchRow(
                    title = stringResource(R.string.settings_developer_info),
                    value = stringResource(R.string.settings_developer_info_value),
                    checked = settings.developerInfo,
                    onCheckedChange = { scope.launch { repo.setDeveloperInfo(it) } },
                )
                SettingRow(
                    title = stringResource(R.string.capabilities_title),
                    icon = YoinSymbols.Info,
                    onClick = { nav.push(Route.Capabilities) },
                )
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

private fun pipelineLabel(id: String): Int = when (id) {
    DngPassthroughPipeline.ID -> R.string.pipeline_dng_passthrough
    else -> R.string.pipeline_unknown
}

private fun gridLabel(g: GridStyle): Int = when (g) {
    GridStyle.THIRDS -> R.string.grid_thirds
    GridStyle.GOLDEN -> R.string.grid_golden
    GridStyle.SQUARE -> R.string.grid_square
}
