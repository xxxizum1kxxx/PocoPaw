package com.izum1k.pocopaw.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn

enum class ColorStyle { EXPRESSIVE, SYSTEM }

enum class FlashMode { OFF, AUTO, ON }

enum class TimerMode(val seconds: Int) { OFF(0), THREE(3), TEN(10) }

enum class AspectRatio(val width: Int, val height: Int) { FOUR_THREE(4, 3), SIXTEEN_NINE(16, 9) }

data class CameraSettings(
    val flash: FlashMode = FlashMode.OFF,
    val timer: TimerMode = TimerMode.OFF,
    val aspect: AspectRatio = AspectRatio.FOUR_THREE,
    val grid: Boolean = false,
    val debugCapture: Boolean = false,
    val lastCameraId: String? = null,
)

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsRepository(context: Context, scope: CoroutineScope) {
    private val store = context.applicationContext.dataStore

    private object Keys {
        val flash = stringPreferencesKey("flash")
        val timer = stringPreferencesKey("timer")
        val aspect = stringPreferencesKey("aspect")
        val grid = booleanPreferencesKey("grid")
        val debugCapture = booleanPreferencesKey("debug_capture")
        val lastCameraId = stringPreferencesKey("last_camera_id")
    }

    val settings: StateFlow<CameraSettings> = store.data.map { p ->
        val d = CameraSettings()
        CameraSettings(
            flash = p[Keys.flash].toEnum(d.flash),
            timer = p[Keys.timer].toEnum(d.timer),
            aspect = p[Keys.aspect].toEnum(d.aspect),
            grid = p[Keys.grid] ?: d.grid,
            debugCapture = p[Keys.debugCapture] ?: d.debugCapture,
            lastCameraId = p[Keys.lastCameraId],
        )
    }.stateIn(scope, SharingStarted.Eagerly, CameraSettings())

    suspend fun setFlash(v: FlashMode) = put(Keys.flash, v.name)
    suspend fun setTimer(v: TimerMode) = put(Keys.timer, v.name)
    suspend fun setAspect(v: AspectRatio) = put(Keys.aspect, v.name)
    suspend fun setGrid(v: Boolean) = put(Keys.grid, v)
    suspend fun setDebugCapture(v: Boolean) = put(Keys.debugCapture, v)
    suspend fun setLastCameraId(v: String) = put(Keys.lastCameraId, v)

    private suspend fun <T> put(key: Preferences.Key<T>, value: T) {
        store.edit { it[key] = value }
    }
}

private inline fun <reified E : Enum<E>> String?.toEnum(default: E): E =
    this?.let { name -> enumValues<E>().firstOrNull { it.name == name } } ?: default
