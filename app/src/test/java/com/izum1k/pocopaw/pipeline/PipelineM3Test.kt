package com.izum1k.pocopaw.pipeline

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineM3Test {

    @Test
    fun flowMethodOrdinalsAndMergeOptionsKeep15FloatArray() {
        assertEquals(0, FlowMethod.BLOCK_MATCH.ordinal)
        assertEquals(1, FlowMethod.SEA_RAFT.ordinal)
        assertEquals(2, FlowMethod.AUTO.ordinal)

        val def = MergeOptions()
        assertEquals(15, def.toArray().size)
        assertEquals(FlowMethod.BLOCK_MATCH, def.flowMethod)
        assertEquals("", def.flowModelPath)

        val seaRaft = MergeOptions(
            flowMethod = FlowMethod.SEA_RAFT,
            flowModelPath = "/data/models/sea_raft.onnx",
        )
        assertEquals(15, seaRaft.toArray().size)
        assertEquals(FlowMethod.SEA_RAFT, seaRaft.flowMethod)
        assertEquals("/data/models/sea_raft.onnx", seaRaft.flowModelPath)
    }

    @Test
    fun tonemapModeOrdinalsAndBurstLookKeep9FloatArray() {
        assertEquals(0, TonemapMode.MERTENS.ordinal)
        assertEquals(1, TonemapMode.HDRNET.ordinal)

        val def = BurstLook()
        assertEquals(9, def.toArray().size)
        assertEquals(TonemapMode.MERTENS, def.tonemapMode)
        assertEquals("", def.hdrnetModelPath)
        assertFalse(def.semanticSegEnabled)
        assertEquals("", def.segModelPath)

        val custom = BurstLook(
            style = "pixel",
            tonemapMode = TonemapMode.HDRNET,
            hdrnetModelPath = "/data/models/hdrnet.onnx",
            semanticSegEnabled = true,
            segModelPath = "/data/models/semantic_seg.onnx",
        )
        assertEquals(9, custom.toArray().size)
        assertEquals(TonemapMode.HDRNET, custom.tonemapMode)
        assertEquals("/data/models/hdrnet.onnx", custom.hdrnetModelPath)
        assertTrue(custom.semanticSegEnabled)
        assertEquals("/data/models/semantic_seg.onnx", custom.segModelPath)
    }

    @Test
    fun modelCatalogExposesAllM3OnnxFilenames() {
        assertEquals("pmrid.onnx", ModelCatalog.PMRID_FILE)
        assertEquals("sea_raft.onnx", ModelCatalog.SEA_RAFT_FILE)
        assertEquals("hdrnet.onnx", ModelCatalog.HDRNET_FILE)
        assertEquals("semantic_seg.onnx", ModelCatalog.SEG_FILE)
    }

    @Test
    fun pipelineOptionsForwardsFlowTonemapAndSemanticSegPreservingNightPreset() {
        val opts = PipelineOptions(
            scene = BurstScene.NIGHT,
            flowMethod = FlowMethod.AUTO,
            flowModelPath = "/files/models/sea_raft.onnx",
            tonemapMode = TonemapMode.HDRNET,
            hdrnetModelPath = "/files/models/hdrnet.onnx",
            semanticSegEnabled = true,
            segModelPath = "/files/models/semantic_seg.onnx",
        )
        val merged = HdrPlusPipeline.mergeOptions(kernelRegression = false, o = opts)
        assertEquals(FlowMethod.AUTO, merged.flowMethod)
        assertEquals("/files/models/sea_raft.onnx", merged.flowModelPath)

        val look = HdrPlusPipeline.finishLook(opts)
        assertEquals("night", look.style)
        assertEquals(TonemapMode.HDRNET, look.tonemapMode)
        assertEquals("/files/models/hdrnet.onnx", look.hdrnetModelPath)
        assertTrue(look.semanticSegEnabled)
        assertEquals("/files/models/semantic_seg.onnx", look.segModelPath)
    }

    @Test
    fun rawMetadataNormalizeFaceBoxesFiltersScoresAndClampsBounds() {
        val faces = listOf(
            RawMetadata.FaceBoxInput(left = 1000, top = 500, right = 2000, bottom = 1500, score = 80),
            RawMetadata.FaceBoxInput(left = 100, top = 100, right = 500, bottom = 500, score = 49), // below MIN_FACE_SCORE=50
            RawMetadata.FaceBoxInput(left = -200, top = 2200, right = 4500, bottom = 1800, score = 90), // out-of-bounds + inverted Y
            RawMetadata.FaceBoxInput(left = 100, top = 100, right = 100, bottom = 500, score = 95), // zero width
        )
        val norm = RawMetadata.normalizeFaceBoxes(faces, activeWidth = 4000, activeHeight = 3000)
        assertArrayEquals(
            floatArrayOf(
                0.25f, 500f / 3000f, 0.50f, 0.50f,
                0.0f, 0.60f, 1.0f, 2200f / 3000f,
            ),
            norm,
            1e-6f,
        )
        assertEquals(0, RawMetadata.normalizeFaceBoxes(faces, activeWidth = 0, activeHeight = 3000).size)
        assertEquals(0, RawMetadata.normalizeFaceBoxes(emptyList<RawMetadata.FaceBoxInput>(), activeWidth = 4000, activeHeight = 3000).size)

        val shifted = listOf(RawMetadata.FaceBoxInput(left = 1016, top = 512, right = 2016, bottom = 1512, score = 85))
        val shiftedNorm = RawMetadata.normalizeFaceBoxes(
            shifted,
            activeWidth = 4000,
            activeHeight = 3000,
            activeLeft = 16,
            activeTop = 12,
        )
        assertArrayEquals(floatArrayOf(0.25f, 500f / 3000f, 0.50f, 0.50f), shiftedNorm, 1e-6f)
    }
}
