package com.izum1k.pocopaw.feature.settings

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.container
import com.izum1k.pocopaw.ui.component.PawPage
import com.izum1k.pocopaw.ui.component.Section
import com.izum1k.pocopaw.ui.component.SwitchRow
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val appContainer = context.container
    val repo = appContainer.settings
    val settings by repo.settings.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val debugPath = remember(appContainer) { appContainer.debugCapturesDir.absolutePath }

    PawPage(title = stringResource(R.string.settings_title)) {
        item {
            Section(title = stringResource(R.string.settings_developer)) {
                SwitchRow(
                    title = stringResource(R.string.settings_debug_capture),
                    value = stringResource(R.string.settings_debug_capture_value, debugPath),
                    icon = YoinSymbols.Info,
                    checked = settings.debugCapture,
                    onCheckedChange = { scope.launch { repo.setDebugCapture(it) } },
                )
            }
        }
    }
}
