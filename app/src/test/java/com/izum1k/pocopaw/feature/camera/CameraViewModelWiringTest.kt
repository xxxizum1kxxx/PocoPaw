package com.izum1k.pocopaw.feature.camera

import com.izum1k.pocopaw.data.AspectRatio
import com.izum1k.pocopaw.data.BurstFrames
import com.izum1k.pocopaw.data.CameraSettings
import com.izum1k.pocopaw.data.DngContent
import com.izum1k.pocopaw.data.KrScaleMode
import com.izum1k.pocopaw.data.LookStyle
import com.izum1k.pocopaw.data.OutputFormat
import com.izum1k.pocopaw.data.Processing
import com.izum1k.pocopaw.data.SceneMode
import com.izum1k.pocopaw.pipeline.AccelBackend
import com.izum1k.pocopaw.pipeline.BurstScene
import com.izum1k.pocopaw.pipeline.DemosaicMethod
import com.izum1k.pocopaw.pipeline.DngPassthroughPipeline
import com.izum1k.pocopaw.pipeline.FlowMethod
import com.izum1k.pocopaw.pipeline.HdrPlusPipeline
import com.izum1k.pocopaw.pipeline.PipelineRegistry
import com.izum1k.pocopaw.pipeline.TonemapMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraViewModelWiringTest {

    @Test
    fun dngOnlyModeSelectsPassthroughAndDisablesMultiFramePipeline() {
        val cfg = CameraSettings(
            output = OutputFormat.DNG,
            processing = Processing.HDR_PLUS_KR,
            look = LookStyle.FILM_WARM,
            pmridEnabled = true,
            tonemapMode = TonemapMode.HDRNET,
        )
        val id = CameraViewModel.pipelineIdFor(cfg, still = false)
        assertEquals(DngPassthroughPipeline.ID, id)
        val pipeline = PipelineRegistry.create(id)
        assertTrue(pipeline is DngPassthroughPipeline)
        assertEquals(1, pipeline.framesPerCapture)
    }

    @Test
    fun nightSceneDisablesManualLookOverrideAndClearsLutPath() {
        val cfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            scene = SceneMode.NIGHT,
            look = LookStyle.CLASSIC_CHROME,
        )
        val look = CameraViewModel.effectiveLook(cfg, flash = false) { id -> "/files/luts/$id.cube" }
        assertEquals("night", look.style)
        assertEquals("", look.lutPath)
        assertEquals(0f, look.lutStrength, 1e-6f)

        val opts = CameraViewModel.buildPipelineOptions(
            cfg = cfg,
            camId = "0",
            lutPathResolver = { id -> "/files/luts/$id.cube" },
        )
        assertEquals(BurstScene.NIGHT, opts.scene)
        assertEquals("", opts.look.lutPath)
        val finished = HdrPlusPipeline.finishLook(opts)
        assertEquals("night", finished.style)
        assertEquals("", finished.lutPath)
        assertEquals(0f, finished.lutStrength, 1e-6f)
    }

    @Test
    fun hdrPlusKrForcesDngContentReferenceAndWiresKrScale() {
        val cfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS_KR,
            dngContent = DngContent.MERGED,
            krScale = KrScaleMode.AUTO,
            pmridEnabled = true,
        )
        assertEquals(HdrPlusPipeline.ID_KR, CameraViewModel.pipelineIdFor(cfg, still = false))
        assertEquals(HdrPlusPipeline.ID, CameraViewModel.pipelineIdFor(cfg, still = true))

        val opts = CameraViewModel.buildPipelineOptions(
            cfg = cfg,
            camId = "0",
            pmridPath = { "/files/models/pmrid.onnx" },
        )
        assertFalse(opts.mergedDng)
        assertEquals(1, opts.krScale)
        assertTrue(opts.krAuto2x)
        // PMRID is classic-path only
        assertFalse(opts.pmridEnabled)
        assertEquals("", opts.pmridModelPath)

        val twoX = CameraViewModel.buildPipelineOptions(
            cfg = cfg.copy(krScale = KrScaleMode.TWO_X),
            camId = "0",
        )
        assertEquals(2, twoX.krScale)
        assertFalse(twoX.krAuto2x)
    }

    @Test
    fun buildPipelineOptionsWiresAllM1ToM5SettingsAndZoomBoost() {
        val cfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            dngContent = DngContent.MERGED,
            processing = Processing.HDR_PLUS,
            demosaicMethod = DemosaicMethod.AMAZE,
            burstFrames = BurstFrames.EIGHT,
            look = LookStyle.FILM_WARM,
            scene = SceneMode.AUTO,
            aspect = AspectRatio.SIXTEEN_NINE,
            ultraHdr = true,
            gpuAcceleration = true,
            accelBackend = AccelBackend.NPU_DSP,
            pmridEnabled = true,
            flowMethod = FlowMethod.SEA_RAFT,
            tonemapMode = TonemapMode.HDRNET,
            semanticSegEnabled = true,
            earlyStopEnabled = true,
        )
        val opts = CameraViewModel.buildPipelineOptions(
            cfg = cfg,
            camId = "0",
            flash = false,
            cropZoom = 1.5f,
            lutPathResolver = { id -> "/files/luts/$id.cube" },
            pmridPath = { "/files/models/pmrid.onnx" },
            flowPath = { "/files/models/sea_raft.onnx" },
            hdrnetPath = { "/files/models/hdrnet.onnx" },
            segPath = { "/files/models/semantic_seg.onnx" },
        )

        assertEquals(14, opts.frames)
        assertTrue(opts.writeJpeg)
        assertTrue(opts.writeDng)
        assertTrue(opts.mergedDng)
        assertTrue(opts.ultraHdr)
        assertTrue(opts.gpu)
        assertEquals(16f / 9f, opts.aspectRatio, 1e-5f)
        assertEquals(DemosaicMethod.AMAZE, opts.demosaic)
        assertEquals("default", opts.look.style)
        assertEquals("/files/luts/film_warm.cube", opts.look.lutPath)
        assertEquals(1f, opts.look.lutStrength, 1e-6f)
        assertTrue(opts.pmridEnabled)
        assertEquals("/files/models/pmrid.onnx", opts.pmridModelPath)
        assertEquals(AccelBackend.NPU_DSP, opts.accelBackend)
        assertEquals(FlowMethod.SEA_RAFT, opts.flowMethod)
        assertEquals("/files/models/sea_raft.onnx", opts.flowModelPath)
        assertEquals(TonemapMode.HDRNET, opts.tonemapMode)
        assertEquals("/files/models/hdrnet.onnx", opts.hdrnetModelPath)
        assertTrue(opts.semanticSegEnabled)
        assertEquals("/files/models/semantic_seg.onnx", opts.segModelPath)
        assertTrue(opts.earlyStopEnabled)

        val merged = HdrPlusPipeline.mergeOptions(kernelRegression = false, o = opts)
        assertEquals(DemosaicMethod.AMAZE, merged.demosaic)
        assertEquals(FlowMethod.SEA_RAFT, merged.flowMethod)
        assertEquals("/files/models/pmrid.onnx", merged.pmridModelPath)
        assertTrue(merged.earlyStopEnabled)

        val finished = HdrPlusPipeline.finishLook(opts)
        assertEquals("/files/luts/film_warm.cube", finished.lutPath)
        assertEquals(TonemapMode.HDRNET, finished.tonemapMode)
        assertEquals("/files/models/hdrnet.onnx", finished.hdrnetModelPath)
        assertTrue(finished.semanticSegEnabled)
        assertEquals("/files/models/semantic_seg.onnx", finished.segModelPath)
    }

    @Test
    fun sessionSettingsOverridesAndClearsWhenSavedDefaultsChange() {
        val saved = CameraSettings()
        val quick = SessionSettings(
            krScale = KrScaleMode.TWO_X,
            look = LookStyle.BW_SILVER,
            tonemapMode = TonemapMode.HDRNET,
            earlyStopEnabled = true,
        )
        val merged = quick.over(saved)
        assertEquals(KrScaleMode.TWO_X, merged.krScale)
        assertEquals(LookStyle.BW_SILVER, merged.look)
        assertEquals(TonemapMode.HDRNET, merged.tonemapMode)
        assertTrue(merged.earlyStopEnabled)

        // Changing saved look clears session look while keeping other session choices
        val nextSaved = saved.copy(look = LookStyle.PIXEL)
        val trimmed = quick.without(saved, nextSaved)
        assertEquals(null, trimmed.look)
        assertEquals(KrScaleMode.TWO_X, trimmed.krScale)
        assertEquals(TonemapMode.HDRNET, trimmed.tonemapMode)
        assertTrue(trimmed.earlyStopEnabled == true)
    }

    @Test
    fun earlyStopEnabledIsGatedByIsEarlyStopEnabledForKrStillAndBracket() {
        val krCfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS_KR,
            scene = SceneMode.AUTO,
            earlyStopEnabled = true,
        )
        assertFalse(krCfg.isEarlyStopEnabled())
        // Even on a tripod where pipelineIdFor falls back to classic HDR+, earlyStopEnabled stays false
        assertEquals(HdrPlusPipeline.ID, CameraViewModel.pipelineIdFor(krCfg, still = true))
        val krOpts = CameraViewModel.buildPipelineOptions(cfg = krCfg, camId = "0")
        assertFalse(krOpts.earlyStopEnabled)

        val bracketCfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS,
            scene = SceneMode.BRACKET,
            earlyStopEnabled = true,
        )
        assertFalse(bracketCfg.isEarlyStopEnabled())
        val bracketOpts = CameraViewModel.buildPipelineOptions(cfg = bracketCfg, camId = "0")
        assertFalse(bracketOpts.earlyStopEnabled)

        val nightCfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS,
            scene = SceneMode.NIGHT,
            earlyStopEnabled = true,
        )
        assertTrue(nightCfg.isEarlyStopEnabled())
        val nightOpts = CameraViewModel.buildPipelineOptions(cfg = nightCfg, camId = "0")
        assertTrue(nightOpts.earlyStopEnabled)
    }

    @Test
    fun expectedFramesMatchesPipelineFramesWithoutResolvingAssets() {
        val dngCfg = CameraSettings(
            output = OutputFormat.DNG,
            burstFrames = BurstFrames.SIXTEEN,
            look = LookStyle.FILM_WARM,
            pmridEnabled = true,
            flowMethod = FlowMethod.SEA_RAFT,
            tonemapMode = TonemapMode.HDRNET,
            semanticSegEnabled = true,
        )
        assertEquals(1, CameraViewModel.expectedFrames(dngCfg, flash = false, cropZoom = 2.0f))
        assertEquals(1, PipelineRegistry.create(CameraViewModel.pipelineIdFor(dngCfg)).framesPerCapture)

        val burstCfg = CameraSettings(
            output = OutputFormat.DNG_JPEG,
            processing = Processing.HDR_PLUS,
            burstFrames = BurstFrames.FOUR,
        )
        val zooms = listOf(1.0f, 1.01f, 1.25f, 1.5f, 2.0f)
        for (flash in listOf(false, true)) {
            for (z in zooms) {
                val expected = CameraViewModel.expectedFrames(burstCfg, flash = flash, cropZoom = z)
                val opts = CameraViewModel.buildPipelineOptions(
                    cfg = burstCfg,
                    camId = "0",
                    flash = flash,
                    cropZoom = z,
                    lutPathResolver = { throw AssertionError("LUT resolver must not be called") },
                    pmridPath = { throw AssertionError("PMRID resolver must not be called") },
                    flowPath = { throw AssertionError("Flow resolver must not be called") },
                    hdrnetPath = { throw AssertionError("HDRnet resolver must not be called") },
                    segPath = { throw AssertionError("Seg resolver must not be called") },
                )
                assertEquals("flash=$flash cropZoom=$z", expected, opts.frames)
            }
        }
    }

    @Test
    fun continuousZoomClampAndCropZoomForHandleNormalAndDegenerateRanges() {
        // Main back lens [1f, 10f]
        assertEquals(1.0f, CameraViewModel.clampContinuousZoom(0.5f, isMainBack = true, zoomLower = 1f, zoomUpper = 10f), 1e-6f)
        assertEquals(1.5f, CameraViewModel.clampContinuousZoom(1.5f, isMainBack = true, zoomLower = 1f, zoomUpper = 10f), 1e-6f)
        assertEquals(2.0f, CameraViewModel.clampContinuousZoom(3.0f, isMainBack = true, zoomLower = 1f, zoomUpper = 10f), 1e-6f)
        assertEquals(1.5f, CameraViewModel.cropZoomFor(isMainBack = true, zoom = 1.5f, zoomUpper = 10f), 1e-6f)
        assertEquals(2.0f, CameraViewModel.cropZoomFor(isMainBack = true, zoom = 3.0f, zoomUpper = 10f), 1e-6f)

        // Ultrawide / non-main lens (isMainBack = false, [1f, 5f])
        assertEquals(3.5f, CameraViewModel.clampContinuousZoom(3.5f, isMainBack = false, zoomLower = 1f, zoomUpper = 5f), 1e-6f)
        assertEquals(5.0f, CameraViewModel.clampContinuousZoom(6.0f, isMainBack = false, zoomLower = 1f, zoomUpper = 5f), 1e-6f)
        assertEquals(1.0f, CameraViewModel.cropZoomFor(isMainBack = false, zoom = 3.5f, zoomUpper = 5f), 1e-6f)

        // Hypothetical range with lower > 2f (must not throw IllegalArgumentException)
        assertEquals(2.5f, CameraViewModel.clampContinuousZoom(1.0f, isMainBack = true, zoomLower = 2.5f, zoomUpper = 4.0f), 1e-6f)
        assertEquals(2.5f, CameraViewModel.clampContinuousZoom(3.0f, isMainBack = true, zoomLower = 2.5f, zoomUpper = 4.0f), 1e-6f)

        // Hypothetical range with upper < 1f (must not throw IllegalArgumentException)
        assertEquals(1.0f, CameraViewModel.clampContinuousZoom(0.5f, isMainBack = true, zoomLower = 0.5f, zoomUpper = 0.8f), 1e-6f)
        assertEquals(1.0f, CameraViewModel.clampContinuousZoom(1.5f, isMainBack = false, zoomLower = 0.5f, zoomUpper = 0.8f), 1e-6f)
        assertEquals(1.0f, CameraViewModel.cropZoomFor(isMainBack = true, zoom = 1.5f, zoomUpper = 0.8f), 1e-6f)

        // Non-finite inputs return fallback / 1f without corrupting state
        assertEquals(1.25f, CameraViewModel.clampContinuousZoom(Float.NaN, isMainBack = true, zoomLower = 1f, zoomUpper = 10f, fallback = 1.25f), 1e-6f)
        assertEquals(1.25f, CameraViewModel.clampContinuousZoom(Float.POSITIVE_INFINITY, isMainBack = true, zoomLower = 1f, zoomUpper = 10f, fallback = 1.25f), 1e-6f)
        assertEquals(1.25f, CameraViewModel.clampContinuousZoom(Float.NEGATIVE_INFINITY, isMainBack = true, zoomLower = 1f, zoomUpper = 10f, fallback = 1.25f), 1e-6f)
        assertEquals(1.0f, CameraViewModel.cropZoomFor(isMainBack = true, zoom = Float.NaN, zoomUpper = 10f), 1e-6f)
        assertEquals(1.0f, CameraViewModel.cropZoomFor(isMainBack = true, zoom = Float.POSITIVE_INFINITY, zoomUpper = 10f), 1e-6f)
    }

    @Test
    fun zslCapturePolicyEnablesAutoPhotoHybridAndReservesHighlightShortFrames() {
        val zslCfg = CameraSettings(zslEnabled = true, scene = SceneMode.AUTO, burstFrames = BurstFrames.EIGHT)

        // Photo Auto with buffered ZSL frames and no highlight clip: full ZSL eligible, hybrid stream disabled so align() scales mixed EVs
        val normal = CameraViewModel.zslCapturePolicy(
            cfg = zslCfg,
            mode = CaptureMode.PHOTO,
            flash = false,
            frames = 8,
            clippedFraction = 0.01f,
            bufferedZsl = 4,
        )
        assertTrue(normal.allowZsl)
        assertEquals(0, normal.reserveShortFrames)
        assertTrue(normal.disableStreamForHybrid)

        // Clipped highlights reserve 2 post-shutter -1 EV frames while keeping ZSL active
        val clipped = CameraViewModel.zslCapturePolicy(
            cfg = zslCfg,
            mode = CaptureMode.PHOTO,
            flash = false,
            frames = 8,
            clippedFraction = 0.12f,
            bufferedZsl = 8,
        )
        assertTrue(clipped.allowZsl)
        assertEquals(2, clipped.reserveShortFrames)
        assertTrue(clipped.disableStreamForHybrid)

        // Cold ring keeps streaming open
        val cold = CameraViewModel.zslCapturePolicy(
            cfg = zslCfg,
            mode = CaptureMode.PHOTO,
            flash = false,
            frames = 8,
            clippedFraction = 0.12f,
            bufferedZsl = 0,
        )
        assertTrue(cold.allowZsl)
        assertEquals(2, cold.reserveShortFrames)
        assertFalse(cold.disableStreamForHybrid)

        // Night, Pro, flash, or ZSL off disables allowZsl
        assertFalse(CameraViewModel.zslCapturePolicy(zslCfg.copy(scene = SceneMode.NIGHT), CaptureMode.PHOTO, false, 8, 0f, 4).allowZsl)
        assertFalse(CameraViewModel.zslCapturePolicy(zslCfg, CaptureMode.PRO, false, 8, 0f, 4).allowZsl)
        assertFalse(CameraViewModel.zslCapturePolicy(zslCfg, CaptureMode.PHOTO, true, 8, 0f, 4).allowZsl)
        assertFalse(CameraViewModel.zslCapturePolicy(zslCfg.copy(zslEnabled = false), CaptureMode.PHOTO, false, 8, 0f, 4).allowZsl)

        // Early stop keeps stream active when requested
        val early = CameraViewModel.zslCapturePolicy(
            cfg = zslCfg.copy(earlyStopEnabled = true),
            mode = CaptureMode.PHOTO,
            flash = false,
            frames = 8,
            clippedFraction = 0f,
            bufferedZsl = 4,
        )
        assertTrue(early.allowZsl)
        assertFalse(early.disableStreamForHybrid)
    }
}
