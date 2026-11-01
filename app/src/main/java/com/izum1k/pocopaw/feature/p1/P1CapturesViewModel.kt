package com.izum1k.pocopaw.feature.p1

import android.content.ContentResolver
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class ExportResult { DONE, FAILED }

data class P1ListState(
    /** Finished bursts per item name. */
    val sets: Map<String, Int> = emptyMap(),
    val bytes: Long = 0,
    /** Null when not exporting. */
    val exportProgress: Float? = null,
    val exportResult: ExportResult? = null,
)

class P1CapturesViewModel(private val store: P1Store, private val resolver: ContentResolver) : ViewModel() {
    private val _state = MutableStateFlow(P1ListState())
    val state: StateFlow<P1ListState> = _state.asStateFlow()

    val rootPath: String get() = store.root.absolutePath

    fun refresh() {
        viewModelScope.launch {
            val (sets, bytes) = withContext(Dispatchers.IO) {
                P1Items.all.associate { it.name to store.sets(it).size } to
                    store.root.walkTopDown().filter { it.isFile }.sumOf { it.length() }
            }
            _state.update { it.copy(sets = sets, bytes = bytes) }
        }
    }

    fun export(target: Uri, items: List<P1Item>) {
        if (_state.value.exportProgress != null) return
        _state.update { it.copy(exportProgress = 0f, exportResult = null) }
        viewModelScope.launch {
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    var shown = 0
                    ZipExporter.export(resolver, target, store.root, items) { p ->
                        val pct = (p * 100).toInt()
                        if (pct != shown) {
                            shown = pct
                            _state.update { it.copy(exportProgress = p) }
                        }
                    }
                }.onFailure { Log.w(TAG, "export failed", it) }.isSuccess
            }
            _state.update { it.copy(exportProgress = null, exportResult = if (ok) ExportResult.DONE else ExportResult.FAILED) }
        }
    }

    fun clear() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { store.clear() }
            consumeResult()
            refresh()
        }
    }

    fun consumeResult() = _state.update { it.copy(exportResult = null) }

    private companion object {
        const val TAG = "PocoPawP1"
    }
}
