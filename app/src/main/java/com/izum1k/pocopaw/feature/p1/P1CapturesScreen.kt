package com.izum1k.pocopaw.feature.p1

import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.gpo.yoin.symbols.YoinSymbols
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.ui.component.PawPage
import com.izum1k.pocopaw.ui.component.Section
import com.izum1k.pocopaw.ui.component.SettingRow
import com.izum1k.pocopaw.ui.navigation.LocalNavigator
import com.izum1k.pocopaw.ui.navigation.Route
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun P1CapturesScreen() {
    val context = LocalContext.current
    val nav = LocalNavigator.current
    val vm: P1CapturesViewModel = viewModel { P1CapturesViewModel(P1Store(context), context.contentResolver) }
    val state by vm.state.collectAsStateWithLifecycle()
    var picking by remember { mutableStateOf(false) }
    var confirmClear by remember { mutableStateOf(false) }
    var chosen by rememberSaveable { mutableStateOf<List<String>>(emptyList()) }
    var exportName by rememberSaveable { mutableStateOf("") }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        if (uri != null) vm.export(uri, P1Items.all.filter { it.name in chosen })
    }

    LifecycleResumeEffect(vm) {
        vm.refresh()
        onPauseOrDispose { }
    }

    val captured = P1Items.all.filter { (state.sets[it.name] ?: 0) > 0 }
    val totalSets = state.sets.values.sum()

    PawPage(title = stringResource(R.string.p1_title)) {
        item {
            Section {
                val progress = state.exportProgress
                SettingRow(
                    title = stringResource(R.string.p1_export),
                    value = when {
                        progress != null -> stringResource(R.string.p1_exporting, (progress * 100).toInt())
                        state.exportResult == ExportResult.DONE -> stringResource(R.string.p1_export_done, exportName)
                        state.exportResult == ExportResult.FAILED -> stringResource(R.string.p1_export_failed)
                        totalSets == 0 -> stringResource(R.string.p1_empty)
                        else -> stringResource(R.string.p1_export_value, totalSets, Formatter.formatShortFileSize(context, state.bytes))
                    },
                    icon = YoinSymbols.Image,
                    enabled = totalSets > 0 && progress == null,
                    onClick = {
                        vm.consumeResult()
                        chosen = captured.map { it.name }
                        picking = true
                    },
                )
                if (progress != null) {
                    LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp))
                }
                SettingRow(
                    title = stringResource(R.string.p1_clear),
                    icon = YoinSymbols.Refresh,
                    enabled = totalSets > 0 && progress == null,
                    onClick = { confirmClear = true },
                )
                Text(
                    stringResource(R.string.p1_adb, vm.rootPath),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                )
            }
        }
        P1Group.entries.forEach { group ->
            item(key = group.name) {
                Section(title = stringResource(group.title)) {
                    Text(
                        stringResource(group.note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                    )
                    P1Items.all.filter { it.group == group }.forEach { item ->
                        val scene = item.scene?.let { stringResource(it) + "\n" }.orEmpty()
                        SettingRow(
                            title = item.name,
                            value = scene + lensLabel(item) + " · " + variantDescription(item.required),
                            onClick = { nav.push(Route.P1Capture(item.name)) },
                            trailing = {
                                Text(
                                    stringResource(R.string.p1_sets, state.sets[item.name] ?: 0),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = if ((state.sets[item.name] ?: 0) > 0) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            },
                        )
                    }
                }
            }
        }
    }

    if (picking) {
        AlertDialog(
            onDismissRequest = { picking = false },
            title = { Text(stringResource(R.string.p1_export_pick)) },
            text = {
                Column(Modifier.verticalScroll(rememberScrollState())) {
                    captured.forEach { item ->
                        val on = item.name in chosen
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .clickable { chosen = if (on) chosen - item.name else chosen + item.name },
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Checkbox(checked = on, onCheckedChange = null)
                            Text(
                                item.name + "  ·  " + stringResource(R.string.p1_sets, state.sets[item.name] ?: 0),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.padding(start = 12.dp),
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    enabled = chosen.isNotEmpty(),
                    onClick = {
                        picking = false
                        exportName = "pocopaw_p1_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".zip"
                        launcher.launch(exportName)
                    },
                ) { Text(stringResource(R.string.p1_export)) }
            },
            dismissButton = { TextButton(onClick = { picking = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            text = { Text(stringResource(R.string.p1_clear_confirm)) },
            confirmButton = {
                TextButton(onClick = {
                    confirmClear = false
                    vm.clear()
                }) { Text(stringResource(R.string.action_delete)) }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}
