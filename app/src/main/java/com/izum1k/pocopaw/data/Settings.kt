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

enum class GridStyle { THIRDS, GOLDEN, SQUARE }

enum class OutputFormat(val available: Boolean) { DNG(true), JPEG(true), DNG_JPEG(true) }

/** Which multi-frame merge renders JPEG output. */
enum class Processing { HDR_PLUS, HDR_PLUS_KR }

enum class BurstFrames(val count: Int) { FOUR(4), EIGHT(8), SIXTEEN(16) }

/** Finishing look of the burst pipeline: the original tuning or the Pixel-like preset. */
enum class LookStyle(val preset: String) { STANDARD("default"), PIXEL("pixel") }

/** What a burst is captured for: one exposure, an exposure bracket, or long motion-metered night frames. */
enum class SceneMode { AUTO, BRACKET, NIGHT }

data class CameraSettings(
    val colorStyle: ColorStyle = ColorStyle.EXPRESSIVE,
    val flash: FlashMode = FlashMode.OFF,
    val timer: TimerMode = TimerMode.OFF,
    val aspect: AspectRatio = AspectRatio.FOUR_THREE,
    val grid: Boolean = false,
    val gridStyle: GridStyle = GridStyle.THIRDS,
    val shutterSound: Boolean = true,
    val haptics: Boolean = true,
    val developerInfo: Boolean = true,
    val output: OutputFormat = OutputFormat.DNG_JPEG,
    val processing: Processing = Processing.HDR_PLUS,
    val burstFrames: BurstFrames = BurstFrames.EIGHT,
    val look: LookStyle = LookStyle.STANDARD,
    val scene: SceneMode = SceneMode.AUTO,
    val ultraHdr: Boolean = false,
    /** Burst processing on the GPU (OpenCL), each step falling back to the CPU on failure. */
    val gpuAcceleration: Boolean = true,
    val lastCameraId: String? = null,
)

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsRepository(context: Context, scope: CoroutineScope) {
    private val store = context.applicationContext.dataStore

    private object Keys {
        val colorStyle = stringPreferencesKey("color_style")
        val flash = stringPreferencesKey("flash")
        val timer = stringPreferencesKey("timer")
        val aspect = stringPreferencesKey("aspect")
        val grid = booleanPreferencesKey("grid")
        val gridStyle = stringPreferencesKey("grid_style")
        val shutterSound = booleanPreferencesKey("shutter_sound")
        val haptics = booleanPreferencesKey("haptics")
        val developerInfo = booleanPreferencesKey("developer_info")
        val output = stringPreferencesKey("output")
        val processing = stringPreferencesKey("processing")
        val burstFrames = stringPreferencesKey("burst_frames")
        val look = stringPreferencesKey("look")
        val scene = stringPreferencesKey("scene")
        val ultraHdr = booleanPreferencesKey("ultra_hdr")
        val gpuAcceleration = booleanPreferencesKey("gpu_acceleration")
        val lastCameraId = stringPreferencesKey("last_camera_id")
    }

    val settings: StateFlow<CameraSettings> = store.data.map { p ->
        val d = CameraSettings()
        CameraSettings(
            colorStyle = p[Keys.colorStyle].toEnum(d.colorStyle),
            flash = p[Keys.flash].toEnum(d.flash),
            timer = p[Keys.timer].toEnum(d.timer),
            aspect = p[Keys.aspect].toEnum(d.aspect),
            grid = p[Keys.grid] ?: d.grid,
            gridStyle = p[Keys.gridStyle].toEnum(d.gridStyle),
            shutterSound = p[Keys.shutterSound] ?: d.shutterSound,
            haptics = p[Keys.haptics] ?: d.haptics,
            developerInfo = p[Keys.developerInfo] ?: d.developerInfo,
            output = p[Keys.output].toEnum(d.output).takeIf { it.available } ?: d.output,
            processing = p[Keys.processing].toEnum(d.processing),
            burstFrames = p[Keys.burstFrames].toEnum(d.burstFrames),
            look = p[Keys.look].toEnum(d.look),
            scene = p[Keys.scene].toEnum(d.scene),
            ultraHdr = p[Keys.ultraHdr] ?: d.ultraHdr,
            gpuAcceleration = p[Keys.gpuAcceleration] ?: d.gpuAcceleration,
            lastCameraId = p[Keys.lastCameraId],
        )
    }.stateIn(scope, SharingStarted.Eagerly, CameraSettings())

    suspend fun setColorStyle(v: ColorStyle) = put(Keys.colorStyle, v.name)
    suspend fun setFlash(v: FlashMode) = put(Keys.flash, v.name)
    suspend fun setTimer(v: TimerMode) = put(Keys.timer, v.name)
    suspend fun setAspect(v: AspectRatio) = put(Keys.aspect, v.name)
    suspend fun setGrid(v: Boolean) = put(Keys.grid, v)
    suspend fun setGridStyle(v: GridStyle) = put(Keys.gridStyle, v.name)
    suspend fun setShutterSound(v: Boolean) = put(Keys.shutterSound, v)
    suspend fun setHaptics(v: Boolean) = put(Keys.haptics, v)
    suspend fun setDeveloperInfo(v: Boolean) = put(Keys.developerInfo, v)
    suspend fun setOutput(v: OutputFormat) = put(Keys.output, v.name)
    suspend fun setProcessing(v: Processing) = put(Keys.processing, v.name)
    suspend fun setBurstFrames(v: BurstFrames) = put(Keys.burstFrames, v.name)
    suspend fun setLook(v: LookStyle) = put(Keys.look, v.name)
    suspend fun setScene(v: SceneMode) = put(Keys.scene, v.name)
    suspend fun setUltraHdr(v: Boolean) = put(Keys.ultraHdr, v)
    suspend fun setGpuAcceleration(v: Boolean) = put(Keys.gpuAcceleration, v)
    suspend fun setLastCameraId(v: String) = put(Keys.lastCameraId, v)

    private suspend fun <T> put(key: Preferences.Key<T>, value: T) {
        store.edit { it[key] = value }
    }
}

private inline fun <reified E : Enum<E>> String?.toEnum(default: E): E =
    this?.let { name -> enumValues<E>().firstOrNull { it.name == name } } ?: default
