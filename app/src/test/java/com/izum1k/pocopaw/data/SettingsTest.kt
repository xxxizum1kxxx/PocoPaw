package com.izum1k.pocopaw.data

import androidx.datastore.core.CorruptionException
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.mutablePreferencesOf
import androidx.datastore.preferences.core.stringPreferencesKey
import com.izum1k.pocopaw.camera.AwbEstimatorMode
import com.izum1k.pocopaw.pipeline.AccelBackend
import com.izum1k.pocopaw.pipeline.DemosaicMethod
import com.izum1k.pocopaw.pipeline.FlowMethod
import com.izum1k.pocopaw.pipeline.TonemapMode
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class SettingsTest {

    @Test
    fun settingsRepositoryUnknownEnumFallback() = runBlocking {
        val dir = createTempDirectory("settings_fallback_test").toFile()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        try {
            val file = File(dir, "corrupt_enums.preferences_pb")
            val store = PreferenceDataStoreFactory.create(scope = scope, produceFile = { file })
            store.edit { prefs ->
                val k = SettingsRepository.Keys
                prefs[k.colorStyle] = "UNKNOWN_COLOR"
                prefs[k.flash] = "STROBE"
                prefs[k.timer] = "SIXTY"
                prefs[k.aspect] = "TWENTY_ONE_NINE"
                prefs[k.gridStyle] = "SPIRAL"
                prefs[k.output] = "HEIF"
                prefs[k.dngContent] = "SYNTHETIC"
                prefs[k.processing] = "MAGIC_AI"
                prefs[k.krScale] = "FOUR_X"
                prefs[k.demosaicMethod] = "VNG"
                prefs[k.burstFrames] = "THIRTY_TWO"
                prefs[k.look] = "CYBERPUNK"
                prefs[k.scene] = "PORTRAIT_X"
                prefs[k.accelBackend] = "QUANTUM"
                prefs[k.awbMode] = "UNKNOWN_AWB"
                prefs[k.flowMethod] = "RAFT_V99"
                prefs[k.tonemapMode] = "REINHARD"
                prefs[k.lastCameraId] = "sentinel_loaded"
            }

            val repo = SettingsRepository(store, scope)
            val decoded = withTimeout(5_000L) {
                repo.settings.first { it.lastCameraId == "sentinel_loaded" }
            }
            val defaults = CameraSettings()

            assertEquals(defaults.colorStyle, decoded.colorStyle)
            assertEquals(defaults.flash, decoded.flash)
            assertEquals(defaults.timer, decoded.timer)
            assertEquals(defaults.aspect, decoded.aspect)
            assertEquals(defaults.gridStyle, decoded.gridStyle)
            assertEquals(defaults.output, decoded.output)
            assertEquals(defaults.dngContent, decoded.dngContent)
            assertEquals(defaults.processing, decoded.processing)
            assertEquals(defaults.krScale, decoded.krScale)
            assertEquals(defaults.demosaicMethod, decoded.demosaicMethod)
            assertEquals(defaults.burstFrames, decoded.burstFrames)
            assertEquals(defaults.look, decoded.look)
            assertEquals(defaults.scene, decoded.scene)
            assertEquals(defaults.accelBackend, decoded.accelBackend)
            assertEquals(defaults.awbMode, decoded.awbMode)
            assertEquals(defaults.flowMethod, decoded.flowMethod)
            assertEquals(defaults.tonemapMode, decoded.tonemapMode)

            // Direct toEnum fallback sanity
            assertEquals(DemosaicMethod.RCD, "NOT_A_DEMOSAIC".toEnum(DemosaicMethod.RCD))
            assertEquals(KrScaleMode.ONE_X, (null as String?).toEnum(KrScaleMode.ONE_X))
            assertEquals(LookStyle.STANDARD, "BAD_LOOK".toEnum(LookStyle.STANDARD))
        } finally {
            scope.cancel()
            dir.deleteRecursively()
        }
    }

    @Test
    fun settingsRepositoryPersistsAllM1ToM5Controls() = runBlocking {
        val dir = createTempDirectory("settings_roundtrip_test").toFile()
        val scope1 = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val scope2 = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        try {
            val file = File(dir, "roundtrip.preferences_pb")
            val store1 = PreferenceDataStoreFactory.create(scope = scope1, produceFile = { file })
            val repo1 = SettingsRepository(store1, scope1)
            repo1.setProcessing(Processing.HDR_PLUS_KR)
            repo1.setKrScale(KrScaleMode.AUTO)
            repo1.setDemosaicMethod(DemosaicMethod.AMAZE)
            repo1.setLook(LookStyle.CLASSIC_CHROME)
            repo1.setAwbMode(AwbEstimatorMode.C5)
            repo1.setPmridEnabled(true)
            repo1.setAccelBackend(AccelBackend.NPU_DSP)
            repo1.setGpuAcceleration(false)
            repo1.setFlowMethod(FlowMethod.SEA_RAFT)
            repo1.setTonemapMode(TonemapMode.HDRNET)
            repo1.setSemanticSegEnabled(true)
            repo1.setZslEnabled(true)
            repo1.setEarlyStopEnabled(true)
            repo1.setRaw10Enabled(true)
            repo1.setLastCameraId("2")

            withTimeout(5_000L) {
                repo1.settings.first { it.lastCameraId == "2" }
            }

            // Copy the flushed .preferences_pb to a new path so store2 deserializes cleanly from disk
            // without colliding with DataStore's active-file registry on the original path.
            val file2 = File(dir, "roundtrip_reopened.preferences_pb")
            file.copyTo(file2, overwrite = true)

            val store2 = PreferenceDataStoreFactory.create(scope = scope2, produceFile = { file2 })
            val repo2 = SettingsRepository(store2, scope2)
            val loaded = withTimeout(5_000L) {
                repo2.settings.first { it.lastCameraId == "2" }
            }

            assertEquals(Processing.HDR_PLUS_KR, loaded.processing)
            assertEquals(KrScaleMode.AUTO, loaded.krScale)
            assertEquals(DemosaicMethod.AMAZE, loaded.demosaicMethod)
            assertEquals(LookStyle.CLASSIC_CHROME, loaded.look)
            assertEquals(AwbEstimatorMode.C5, loaded.awbMode)
            assertTrue(loaded.pmridEnabled)
            assertEquals(AccelBackend.NPU_DSP, loaded.accelBackend)
            assertFalse(loaded.gpuAcceleration)
            assertEquals(FlowMethod.SEA_RAFT, loaded.flowMethod)
            assertEquals(TonemapMode.HDRNET, loaded.tonemapMode)
            assertTrue(loaded.semanticSegEnabled)
            assertTrue(loaded.zslEnabled)
            assertTrue(loaded.earlyStopEnabled)
            assertTrue(loaded.raw10Enabled)
        } finally {
            scope1.cancel()
            scope2.cancel()
            dir.deleteRecursively()
        }
    }

    @Test
    fun dngOnlyModeDisablesMultiFrameAndLookOptions() {
        val dngOnly = CameraSettings(
            output = OutputFormat.DNG,
            processing = Processing.HDR_PLUS,
            scene = SceneMode.AUTO,
            look = LookStyle.FILM_WARM,
            pmridEnabled = true,
            earlyStopEnabled = true,
        )
        assertFalse(dngOnly.isBurstMode)
        assertFalse(dngOnly.isLookEnabled())
        assertFalse(dngOnly.isDemosaicEnabled())
        assertFalse(dngOnly.isPmridEnabled())
        assertFalse(dngOnly.isKrScaleEnabled())
        assertFalse(dngOnly.isEarlyStopEnabled())
    }

    @Test
    fun nightSceneDisablesManualLookOverride() {
        val night = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS,
            scene = SceneMode.NIGHT,
            look = LookStyle.BW_SILVER,
        )
        assertTrue(night.isBurstMode)
        assertFalse(night.isLookEnabled())
        assertTrue(night.isEarlyStopEnabled())
        assertTrue(night.isDemosaicEnabled())
    }

    @Test
    fun hdrPlusKrForcesDngContentReference() {
        val krWithMergedPref = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS_KR,
            dngContent = DngContent.MERGED,
            krScale = KrScaleMode.TWO_X,
        )
        assertEquals(DngContent.REFERENCE, krWithMergedPref.effectiveDngContent())
        assertTrue(krWithMergedPref.isKrScaleEnabled())
        assertFalse(krWithMergedPref.isDemosaicEnabled())
        assertFalse(krWithMergedPref.isPmridEnabled())
        assertFalse(krWithMergedPref.isEarlyStopEnabled())

        val classic = krWithMergedPref.copy(processing = Processing.HDR_PLUS)
        assertEquals(DngContent.MERGED, classic.effectiveDngContent())
        assertFalse(classic.isKrScaleEnabled())
        assertTrue(classic.isDemosaicEnabled())
        assertTrue(classic.isPmridEnabled())
    }

    @Test
    fun effectiveGpuAccelerationRespectsAccelBackendPolicy() {
        assertFalse(CameraSettings(gpuAcceleration = true, accelBackend = AccelBackend.CPU).effectiveGpuAcceleration())
        assertTrue(CameraSettings(gpuAcceleration = false, accelBackend = AccelBackend.GPU_OPENCL).effectiveGpuAcceleration())
        assertTrue(CameraSettings(gpuAcceleration = true, accelBackend = AccelBackend.AUTO).effectiveGpuAcceleration())
        assertFalse(CameraSettings(gpuAcceleration = false, accelBackend = AccelBackend.AUTO).effectiveGpuAcceleration())
        assertTrue(CameraSettings(gpuAcceleration = true, accelBackend = AccelBackend.NPU_DSP).effectiveGpuAcceleration())
        assertFalse(CameraSettings(gpuAcceleration = false, accelBackend = AccelBackend.NPU_DSP).effectiveGpuAcceleration())
    }

    @Test
    fun wrongTypeDataStoreKeysRecoverDefaultsWithoutWedgingSettingsFlow() = runBlocking {
        val defaults = CameraSettings()
        val directDecoded = decodeCameraSettings(
            mutablePreferencesOf(
                intPreferencesKey("color_style") to 42,
                booleanPreferencesKey("demosaic_method") to true,
                floatPreferencesKey("burst_frames") to 3.5f,
                longPreferencesKey("last_camera_id") to 99L,
                stringPreferencesKey("grid") to "true",
                intPreferencesKey("gpu_acceleration") to 1,
                stringPreferencesKey("early_stop_enabled") to "yes",
            ),
        )
        assertEquals(defaults, directDecoded)

        val dir = createTempDirectory("settings_wrong_type_test").toFile()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        try {
            val file = File(dir, "wrong_type.preferences_pb")
            val store = PreferenceDataStoreFactory.create(
                corruptionHandler = settingsCorruptionHandler(),
                scope = scope,
                produceFile = { file },
            )
            val repo = SettingsRepository(store, scope)

            store.edit { prefs ->
                prefs[intPreferencesKey("color_style")] = 99
                prefs[booleanPreferencesKey("demosaic_method")] = false
                prefs[floatPreferencesKey("burst_frames")] = 12.0f
                prefs[longPreferencesKey("last_camera_id")] = 777L
                prefs[stringPreferencesKey("grid")] = "not_a_boolean"
                prefs[intPreferencesKey("gpu_acceleration")] = 0
                prefs[SettingsRepository.Keys.look] = LookStyle.FILM_WARM.name
            }

            val recovered = withTimeout(5_000L) {
                repo.settings.first { it.look == LookStyle.FILM_WARM }
            }
            assertEquals(defaults.colorStyle, recovered.colorStyle)
            assertEquals(defaults.demosaicMethod, recovered.demosaicMethod)
            assertEquals(defaults.burstFrames, recovered.burstFrames)
            assertEquals(null, recovered.lastCameraId)
            assertEquals(defaults.grid, recovered.grid)
            assertEquals(defaults.gpuAcceleration, recovered.gpuAcceleration)

            // StateFlow must remain alive for subsequent writes
            repo.setLastCameraId("0")
            val updated = withTimeout(5_000L) {
                repo.settings.first { it.lastCameraId == "0" }
            }
            assertEquals("0", updated.lastCameraId)
        } finally {
            scope.cancel()
            dir.deleteRecursively()
        }
    }

    @Test
    fun corruptPreferencesPbHandledCleanlyWithCorruptionHandlerAndCatch() = runBlocking {
        val dir = createTempDirectory("settings_corrupt_pb_test").toFile()
        val scope1 = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val scope2 = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        try {
            val file = File(dir, "corrupt.preferences_pb")
            file.writeBytes(byteArrayOf(0xFF.toByte(), 0x00, 0x13, 0x37, 0xDE.toByte(), 0xAD.toByte()))

            val store = PreferenceDataStoreFactory.create(
                corruptionHandler = settingsCorruptionHandler(),
                scope = scope1,
                produceFile = { file },
            )
            val repo = SettingsRepository(store, scope1)

            // Store resets corrupt protobuf via ReplaceFileCorruptionHandler and stays writable
            repo.setLastCameraId("recovered_0")
            val loaded = withTimeout(5_000L) {
                repo.settings.first { it.lastCameraId == "recovered_0" }
            }
            assertEquals(CameraSettings().copy(lastCameraId = "recovered_0"), loaded)

            // Even if a DataStore emits CorruptionException directly, SettingsRepository.settings catches it
            val throwingStore = object : DataStore<Preferences> {
                override val data: Flow<Preferences> = flow {
                    throw CorruptionException("corrupt protobuf payload")
                }

                override suspend fun updateData(transform: suspend (t: Preferences) -> Preferences): Preferences =
                    throw UnsupportedOperationException()
            }
            val catchRepo = SettingsRepository(throwingStore, scope2)
            assertEquals(CameraSettings(), catchRepo.settings.value)
        } finally {
            scope1.cancel()
            scope2.cancel()
            dir.deleteRecursively()
        }
    }
}
