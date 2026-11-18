package com.izum1k.pocopaw.data

import android.content.Context
import androidx.datastore.core.CorruptionException
import androidx.datastore.core.DataStore
import androidx.datastore.core.handlers.ReplaceFileCorruptionHandler
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.izum1k.pocopaw.camera.AwbEstimatorMode
import com.izum1k.pocopaw.pipeline.AccelBackend
import com.izum1k.pocopaw.pipeline.DemosaicMethod
import com.izum1k.pocopaw.pipeline.FlowMethod
import com.izum1k.pocopaw.pipeline.TonemapMode
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import java.io.IOException

enum class ColorStyle { EXPRESSIVE, SYSTEM }

enum class FlashMode { OFF, AUTO, ON }

enum class TimerMode(val seconds: Int) { OFF(0), THREE(3), TEN(10) }

enum class AspectRatio(val width: Int, val height: Int) { FOUR_THREE(4, 3), SIXTEEN_NINE(16, 9) }

enum class GridStyle { THIRDS, GOLDEN, SQUARE }

enum class OutputFormat(val available: Boolean) { DNG(true), JPEG(true), DNG_JPEG(true) }

/** What the DNG of a DNG + JPEG burst holds: the merged frames, or the reference frame as captured. */
enum class DngContent { MERGED, REFERENCE }

/** Which multi-frame merge renders JPEG output. */
enum class Processing { HDR_PLUS, HDR_PLUS_KR }

/** Kernel-regression super-resolution output scale mode (`1x`, `2x`, or `subpixel_diversity` Auto). */
enum class KrScaleMode(val scale: Int, val auto2x: Boolean) {
    ONE_X(1, false),
    TWO_X(2, false),
    AUTO(1, true),
}

enum class BurstFrames(val count: Int) { FOUR(4), EIGHT(8), SIXTEEN(16) }

/** Finishing look of the burst pipeline: preset tuning plus optional bundled 3D `.cube` LUT. */
enum class LookStyle(val preset: String, val lutId: String? = null) {
    STANDARD("default"),
    PIXEL("pixel"),
    FILM_WARM("default", "film_warm"),
    CLASSIC_CHROME("default", "classic_chrome"),
    BW_SILVER("default", "bw_silver"),
}

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
    val dngContent: DngContent = DngContent.MERGED,
    val processing: Processing = Processing.HDR_PLUS,
    val krScale: KrScaleMode = KrScaleMode.ONE_X,
    val demosaicMethod: DemosaicMethod = DemosaicMethod.RCD,
    val burstFrames: BurstFrames = BurstFrames.EIGHT,
    val look: LookStyle = LookStyle.STANDARD,
    val scene: SceneMode = SceneMode.AUTO,
    val ultraHdr: Boolean = false,
    /** Burst processing on the GPU (OpenCL), each step falling back to the CPU on failure. */
    val gpuAcceleration: Boolean = true,
    val accelBackend: AccelBackend = AccelBackend.AUTO,
    val awbMode: AwbEstimatorMode = AwbEstimatorMode.HAL,
    val pmridEnabled: Boolean = false,
    val flowMethod: FlowMethod = FlowMethod.BLOCK_MATCH,
    val tonemapMode: TonemapMode = TonemapMode.MERTENS,
    val semanticSegEnabled: Boolean = false,
    val lastCameraId: String? = null,
    val zslEnabled: Boolean = false,
    val earlyStopEnabled: Boolean = false,
    val raw10Enabled: Boolean = false,
) {
    val isBurstMode: Boolean get() = output != OutputFormat.DNG

    fun effectiveDngContent(): DngContent =
        if (processing == Processing.HDR_PLUS_KR) DngContent.REFERENCE else dngContent

    fun isLookEnabled(): Boolean = isBurstMode && scene != SceneMode.NIGHT

    fun isDemosaicEnabled(): Boolean = isBurstMode && processing == Processing.HDR_PLUS

    fun isPmridEnabled(): Boolean = isBurstMode && processing == Processing.HDR_PLUS

    fun isKrScaleEnabled(): Boolean = isBurstMode && processing == Processing.HDR_PLUS_KR

    fun isEarlyStopEnabled(): Boolean = isBurstMode && processing == Processing.HDR_PLUS && scene != SceneMode.BRACKET

    fun effectiveGpuAcceleration(): Boolean = when (accelBackend) {
        AccelBackend.CPU -> false
        AccelBackend.GPU_OPENCL -> true
        AccelBackend.AUTO, AccelBackend.NPU_DSP -> gpuAcceleration
    }
}

internal fun settingsCorruptionHandler(): ReplaceFileCorruptionHandler<Preferences> =
    ReplaceFileCorruptionHandler { emptyPreferences() }

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(
    name = "settings",
    corruptionHandler = settingsCorruptionHandler(),
)

class SettingsRepository internal constructor(
    private val store: DataStore<Preferences>,
    scope: CoroutineScope,
) {
    constructor(context: Context, scope: CoroutineScope) :
        this(context.applicationContext.dataStore, scope)

    internal object Keys {
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
        val dngContent = stringPreferencesKey("dng_content")
        val processing = stringPreferencesKey("processing")
        val krScale = stringPreferencesKey("kr_scale")
        val demosaicMethod = stringPreferencesKey("demosaic_method")
        val burstFrames = stringPreferencesKey("burst_frames")
        val look = stringPreferencesKey("look")
        val scene = stringPreferencesKey("scene")
        val ultraHdr = booleanPreferencesKey("ultra_hdr")
        val gpuAcceleration = booleanPreferencesKey("gpu_acceleration")
        val accelBackend = stringPreferencesKey("accel_backend")
        val awbMode = stringPreferencesKey("awb_mode")
        val pmridEnabled = booleanPreferencesKey("pmrid_enabled")
        val flowMethod = stringPreferencesKey("flow_method")
        val tonemapMode = stringPreferencesKey("tonemap_mode")
        val semanticSegEnabled = booleanPreferencesKey("semantic_seg_enabled")
        val lastCameraId = stringPreferencesKey("last_camera_id")
        val zslEnabled = booleanPreferencesKey("zsl_enabled")
        val earlyStopEnabled = booleanPreferencesKey("early_stop_enabled")
        val raw10Enabled = booleanPreferencesKey("raw10_enabled")
    }

    val settings: StateFlow<CameraSettings> = store.data
        .catch { e ->
            if (e is IOException || e is CorruptionException) emit(emptyPreferences()) else throw e
        }
        .map(::decodeCameraSettings)
        .stateIn(scope, SharingStarted.Eagerly, CameraSettings())

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
    suspend fun setDngContent(v: DngContent) = put(Keys.dngContent, v.name)
    suspend fun setProcessing(v: Processing) = put(Keys.processing, v.name)
    suspend fun setKrScale(v: KrScaleMode) = put(Keys.krScale, v.name)
    suspend fun setDemosaicMethod(v: DemosaicMethod) = put(Keys.demosaicMethod, v.name)
    suspend fun setBurstFrames(v: BurstFrames) = put(Keys.burstFrames, v.name)
    suspend fun setLook(v: LookStyle) = put(Keys.look, v.name)
    suspend fun setScene(v: SceneMode) = put(Keys.scene, v.name)
    suspend fun setUltraHdr(v: Boolean) = put(Keys.ultraHdr, v)
    suspend fun setGpuAcceleration(v: Boolean) = put(Keys.gpuAcceleration, v)
    suspend fun setAccelBackend(v: AccelBackend) = put(Keys.accelBackend, v.name)
    suspend fun setAwbMode(v: AwbEstimatorMode) = put(Keys.awbMode, v.name)
    suspend fun setPmridEnabled(v: Boolean) = put(Keys.pmridEnabled, v)
    suspend fun setFlowMethod(v: FlowMethod) = put(Keys.flowMethod, v.name)
    suspend fun setTonemapMode(v: TonemapMode) = put(Keys.tonemapMode, v.name)
    suspend fun setSemanticSegEnabled(v: Boolean) = put(Keys.semanticSegEnabled, v)
    suspend fun setLastCameraId(v: String) = put(Keys.lastCameraId, v)
    suspend fun setZslEnabled(v: Boolean) = put(Keys.zslEnabled, v)
    suspend fun setEarlyStopEnabled(v: Boolean) = put(Keys.earlyStopEnabled, v)
    suspend fun setRaw10Enabled(v: Boolean) = put(Keys.raw10Enabled, v)

    private suspend fun <T> put(key: Preferences.Key<T>, value: T) {
        store.edit { it[key] = value }
    }
}

private fun Preferences.str(key: Preferences.Key<String>): String? = asMap()[key] as? String

private fun Preferences.bool(key: Preferences.Key<Boolean>): Boolean? = asMap()[key] as? Boolean

internal fun decodeCameraSettings(p: Preferences): CameraSettings {
    val d = CameraSettings()
    val k = SettingsRepository.Keys
    return CameraSettings(
        colorStyle = p.str(k.colorStyle).toEnum(d.colorStyle),
        flash = p.str(k.flash).toEnum(d.flash),
        timer = p.str(k.timer).toEnum(d.timer),
        aspect = p.str(k.aspect).toEnum(d.aspect),
        grid = p.bool(k.grid) ?: d.grid,
        gridStyle = p.str(k.gridStyle).toEnum(d.gridStyle),
        shutterSound = p.bool(k.shutterSound) ?: d.shutterSound,
        haptics = p.bool(k.haptics) ?: d.haptics,
        developerInfo = p.bool(k.developerInfo) ?: d.developerInfo,
        output = p.str(k.output).toEnum(d.output).takeIf { it.available } ?: d.output,
        dngContent = p.str(k.dngContent).toEnum(d.dngContent),
        processing = p.str(k.processing).toEnum(d.processing),
        krScale = p.str(k.krScale).toEnum(d.krScale),
        demosaicMethod = p.str(k.demosaicMethod).toEnum(d.demosaicMethod),
        burstFrames = p.str(k.burstFrames).toEnum(d.burstFrames),
        look = p.str(k.look).toEnum(d.look),
        scene = p.str(k.scene).toEnum(d.scene),
        ultraHdr = p.bool(k.ultraHdr) ?: d.ultraHdr,
        gpuAcceleration = p.bool(k.gpuAcceleration) ?: d.gpuAcceleration,
        accelBackend = p.str(k.accelBackend).toEnum(d.accelBackend),
        awbMode = p.str(k.awbMode).toEnum(d.awbMode),
        pmridEnabled = p.bool(k.pmridEnabled) ?: d.pmridEnabled,
        flowMethod = p.str(k.flowMethod).toEnum(d.flowMethod),
        tonemapMode = p.str(k.tonemapMode).toEnum(d.tonemapMode),
        semanticSegEnabled = p.bool(k.semanticSegEnabled) ?: d.semanticSegEnabled,
        lastCameraId = p.str(k.lastCameraId),
        zslEnabled = p.bool(k.zslEnabled) ?: d.zslEnabled,
        earlyStopEnabled = p.bool(k.earlyStopEnabled) ?: d.earlyStopEnabled,
        raw10Enabled = p.bool(k.raw10Enabled) ?: d.raw10Enabled,
    )
}

internal inline fun <reified E : Enum<E>> String?.toEnum(default: E): E =
    this?.let { name -> enumValues<E>().firstOrNull { it.name == name } } ?: default
