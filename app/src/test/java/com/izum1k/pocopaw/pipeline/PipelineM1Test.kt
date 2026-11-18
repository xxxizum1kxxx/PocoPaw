package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.ManualExposure
import com.izum1k.pocopaw.data.LookStyle
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class PipelineM1Test {

    @Test
    fun mergeOptionsSerializesKrScaleAuto2xAndSnrThreshold() {
        val def = MergeOptions()
        assertEquals(1, def.krScale)
        assertEquals(false, def.krAuto2x)
        assertEquals(14.0f, def.krSnrThresholdDb, 1e-6f)
        assertEquals(15, def.toArray().size)
        assertArrayEquals(
            floatArrayOf(0f, 0f, 0f, 1f, DemosaicMethod.RCD.ordinal.toFloat(), 0f, 64f, 2f, 0f, 0f, 0f, 0f, 1f, 0f, 14.0f),
            def.toArray(),
            1e-6f,
        )

        val custom = MergeOptions(
            kernelRegression = true,
            demosaic = DemosaicMethod.AMAZE,
            frequencyMerge = true,
            lca = LensLca(0.001f, -0.002f, -0.001f, 0.002f),
            krScale = 2,
            krAuto2x = true,
            krSnrThresholdDb = 16.5f,
        )
        assertArrayEquals(
            floatArrayOf(1f, 0f, 0f, 1f, DemosaicMethod.AMAZE.ordinal.toFloat(), 1f, 64f, 2f, 0.001f, -0.002f, -0.001f, 0.002f, 2f, 1f, 16.5f),
            custom.toArray(),
            1e-6f,
        )
    }

    @Test
    fun burstLookPreservesLutDefaultsAndCustomValues() {
        val def = BurstLook()
        assertEquals("default", def.style)
        assertEquals("", def.lutPath)
        assertEquals(1f, def.lutStrength, 1e-6f)
        assertEquals(9, def.toArray().size)

        val graded = BurstLook(style = "pixel", lutPath = "/data/luts/film_warm.cube", lutStrength = 0.75f)
        assertEquals("pixel", graded.style)
        assertEquals("/data/luts/film_warm.cube", graded.lutPath)
        assertEquals(0.75f, graded.lutStrength, 1e-6f)
    }

    @Test
    fun pipelineOptionsForwardsKrFieldsIntoMergeOptions() {
        val opts = PipelineOptions(
            scene = BurstScene.NIGHT,
            lca = LensLca(0.001f, 0f, -0.001f, 0f),
            krScale = 2,
            krAuto2x = true,
            krSnrThresholdDb = 15.0f,
        )
        val merged = HdrPlusPipeline.mergeOptions(kernelRegression = true, o = opts)
        assertEquals(true, merged.kernelRegression)
        assertEquals(true, merged.frequencyMerge)
        assertEquals(opts.lca, merged.lca)
        assertEquals(2, merged.krScale)
        assertEquals(true, merged.krAuto2x)
        assertEquals(15.0f, merged.krSnrThresholdDb, 1e-6f)
    }

    @Test
    fun openStreamEnablesAutoAndNightAndRejectsBracketPerFrameKrAndSingle() {
        val autoExp = ExposurePlan.Auto()
        val manualExp = ExposurePlan.Manual(ManualExposure(iso = 100, exposureNs = 10_000_000L))
        val perFrameExp = ExposurePlan.PerFrame(
            listOf(
                ManualExposure(iso = 100, exposureNs = 10_000_000L),
                ManualExposure(iso = 100, exposureNs = 40_000_000L),
            ),
        )

        HdrPlusPipeline("auto", kernelRegression = false, PipelineOptions(frames = 8, scene = BurstScene.AUTO))
            .openStream(autoExp).also { stream ->
                assertNotNull("AUTO scene should open stream", stream)
                stream?.close()
            }

        HdrPlusPipeline("night", kernelRegression = false, PipelineOptions(frames = 8, scene = BurstScene.NIGHT))
            .openStream(manualExp).also { stream ->
                assertNotNull("NIGHT scene should open stream", stream)
                stream?.close()
            }

        assertNull(
            "BRACKET scene must not stream",
            HdrPlusPipeline("bracket", kernelRegression = false, PipelineOptions(frames = 8, scene = BurstScene.BRACKET)).openStream(autoExp),
        )
        assertNull(
            "PerFrame exposure must not stream",
            HdrPlusPipeline("night_pf", kernelRegression = false, PipelineOptions(frames = 8, scene = BurstScene.NIGHT)).openStream(perFrameExp),
        )
        assertNull(
            "KR pipeline must not stream",
            HdrPlusPipeline("kr", kernelRegression = true, PipelineOptions(frames = 8, scene = BurstScene.NIGHT)).openStream(autoExp),
        )
        assertNull(
            "Single frame must not stream",
            HdrPlusPipeline("single", kernelRegression = false, PipelineOptions(frames = 1, scene = BurstScene.NIGHT)).openStream(autoExp),
        )
    }

    @Test
    fun lutCatalogExposesExpectedBundledPresets() {
        assertEquals(listOf("film_warm", "classic_chrome", "bw_silver"), LutCatalog.presets.map { it.id })
        assertEquals(listOf("film_warm.cube", "classic_chrome.cube", "bw_silver.cube"), LutCatalog.presets.map { it.fileName })
    }

    @Test
    fun pipelineOptionsForwardsDemosaicAndNightStripsManualLut() {
        val opts = PipelineOptions(
            demosaic = DemosaicMethod.AMAZE,
            scene = BurstScene.NIGHT,
            look = BurstLook(style = "default", lutPath = "/files/luts/film_warm.cube", lutStrength = 1f),
        )
        val merged = HdrPlusPipeline.mergeOptions(kernelRegression = false, o = opts)
        assertEquals(DemosaicMethod.AMAZE, merged.demosaic)

        val nightLook = HdrPlusPipeline.finishLook(opts)
        assertEquals("night", nightLook.style)
        assertEquals("", nightLook.lutPath)
        assertEquals(0f, nightLook.lutStrength, 1e-6f)

        assertEquals(
            listOf(null, null, "film_warm", "classic_chrome", "bw_silver"),
            LookStyle.entries.map { it.lutId },
        )
    }
}
