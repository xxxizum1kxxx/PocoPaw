package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.ManualExposure
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class IndigoHdrPlusPipelineTest {

    @Test
    fun registryRoutesDefaultAndLegacyHdrPlusToLockedIndigoPipeline() {
        assertEquals(IndigoHdrPlusPipeline.ID, PipelineRegistry.defaultId)
        assertTrue(PipelineRegistry.create(null) is IndigoHdrPlusPipeline)
        assertTrue(PipelineRegistry.create(IndigoHdrPlusPipeline.ID) is IndigoHdrPlusPipeline)
        assertTrue(PipelineRegistry.create(HdrPlusPipeline.ID) is IndigoHdrPlusPipeline)
        assertTrue(PipelineRegistry.create(DngPassthroughPipeline.ID) is DngPassthroughPipeline)
    }

    @Test
    fun framesPerCaptureLocksToEightOrSingleFlashFrame() {
        val normal = IndigoHdrPlusPipeline(PipelineOptions())
        assertEquals(8, normal.framesPerCapture)

        // Even if caller asks for 4 or 12, multi-frame Indigo locks to 8 frames
        assertEquals(8, IndigoHdrPlusPipeline(PipelineOptions(frames = 4)).framesPerCapture)
        assertEquals(8, IndigoHdrPlusPipeline(PipelineOptions(frames = 12)).framesPerCapture)

        // Single-frame flash capture (frames = 1) stays 1 and skips BurstStream
        val flashSingle = IndigoHdrPlusPipeline(PipelineOptions(frames = 1))
        assertEquals(1, flashSingle.framesPerCapture)
        assertNull(flashSingle.openStream(ExposurePlan.Auto()))
    }

    @Test
    fun bracketedPerFrameExposureSkipsStreamingSoShortestReferenceCanBePicked() {
        val pipe = IndigoHdrPlusPipeline(PipelineOptions())
        val bracket = ExposurePlan.PerFrame(
            listOf(ManualExposure(100, 8_000_000L), ManualExposure(100, 32_000_000L))
        )
        assertNull(pipe.openStream(bracket))
    }

    @Test
    fun defaultLookAndPipelineOptionsMatchIndigoSpec() {
        val opts = PipelineOptions()
        assertTrue(opts.gpu)
        assertEquals(false, opts.ultraHdr)
        val look = BurstLook()
        assertEquals("indigo", look.style)
        assertEquals(0.20f, look.targetKey, 1e-6f)
        assertEquals(5.0f, look.compression, 1e-6f)
        assertEquals(0.20f, look.contrast, 1e-6f)
        assertEquals(1.08f, look.saturation, 1e-6f)
        assertEquals(0.35f, look.sharpen, 1e-6f)
        assertEquals(0.85f, look.sharpenRadius, 1e-6f)
        assertEquals(0.0f, look.localContrast, 1e-6f)
    }
}
