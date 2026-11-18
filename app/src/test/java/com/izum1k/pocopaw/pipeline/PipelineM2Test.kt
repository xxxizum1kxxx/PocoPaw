package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.AwbEstimatorMode
import com.izum1k.pocopaw.data.CameraSettings
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class PipelineM2Test {

    @Test
    fun mergeOptionsKeeps15FloatArrayAndExposesPmridAndAccelFields() {
        val def = MergeOptions()
        assertEquals(15, def.toArray().size)
        assertFalse(def.pmridEnabled)
        assertEquals(0.05f, def.pmridLowNoiseCutoff, 1e-6f)
        assertEquals("", def.pmridModelPath)
        assertEquals(AccelBackend.AUTO, def.accelBackend)

        val custom = MergeOptions(
            pmridEnabled = true,
            pmridLowNoiseCutoff = 0.03f,
            pmridModelPath = "/data/models/pmrid.onnx",
            accelBackend = AccelBackend.NPU_DSP,
        )
        assertEquals(15, custom.toArray().size)
        assertTrue(custom.pmridEnabled)
        assertEquals(0.03f, custom.pmridLowNoiseCutoff, 1e-6f)
        assertEquals("/data/models/pmrid.onnx", custom.pmridModelPath)
        assertEquals(AccelBackend.NPU_DSP, custom.accelBackend)
    }

    @Test
    fun pipelineOptionsForwardsPmridAndAccelIntoMergeOptions() {
        val opts = PipelineOptions(
            pmridEnabled = true,
            pmridLowNoiseCutoff = 0.04f,
            pmridModelPath = "/files/models/pmrid.onnx",
            accelBackend = AccelBackend.GPU_OPENCL,
        )
        val merged = HdrPlusPipeline.mergeOptions(kernelRegression = false, o = opts)
        assertTrue(merged.pmridEnabled)
        assertEquals(0.04f, merged.pmridLowNoiseCutoff, 1e-6f)
        assertEquals("/files/models/pmrid.onnx", merged.pmridModelPath)
        assertEquals(AccelBackend.GPU_OPENCL, merged.accelBackend)
    }

    @Test
    fun accelBackendCoercesInvalidOrdinalsSafely() {
        assertEquals(AccelBackend.AUTO, AccelBackend.fromOrdinal(0))
        assertEquals(AccelBackend.CPU, AccelBackend.fromOrdinal(1))
        assertEquals(AccelBackend.GPU_OPENCL, AccelBackend.fromOrdinal(2))
        assertEquals(AccelBackend.NPU_DSP, AccelBackend.fromOrdinal(3))
        assertEquals(AccelBackend.AUTO, AccelBackend.fromOrdinal(-1))
        assertEquals(AccelBackend.AUTO, AccelBackend.fromOrdinal(99))
    }

    @Test
    fun accelCapabilitiesParsesProbeReport() {
        val report = """
            opencl gpu: QUALCOMM Adreno(TM) 725
            onnx runtime: 1.31.1
            ort providers: NnapiExecutionProvider, XnnpackExecutionProvider, CPUExecutionProvider
            nnapi ep: available
            qnn ep: QNN unavailable (proprietary Qualcomm QNN SDK not bundled; NPU_DSP uses NNAPI -> CPU fallback)
            ml threads: 4 (big+prime cpus 7,6,5,4)
            cdsp fastrpc: available (arch 0x69, HVX 4, unsigned PD 1)
            hexagon sdk boundary: FastRPC probed; custom cDSP skel compilation requires proprietary Hexagon SDK (unavailable)
        """.trimIndent()
        val cap = AccelCapabilities.parse(report)
        assertTrue(cap.openclAvailable)
        assertEquals("QUALCOMM Adreno(TM) 725", cap.openclDevice)
        assertTrue(cap.ortAvailable)
        assertEquals("1.31.1", cap.ortVersion)
        assertEquals(listOf("NnapiExecutionProvider", "XnnpackExecutionProvider", "CPUExecutionProvider"), cap.ortProviders)
        assertTrue(cap.nnapiAvailable)
        assertFalse(cap.qnnAvailable)
        assertTrue(cap.cdspFastRpcAvailable)
        assertEquals(4, cap.mlThreads)
        assertEquals(listOf(7, 6, 5, 4), cap.bigPrimeCpuIds)
        assertTrue(cap.hexagonSdkBoundary.contains("Hexagon SDK"))
    }

    @Test
    fun ortConfigDetectsBigPrimeCpusAndFormatsExactWorkerAffinities() {
        val tmpRoot = createTempDirectory("cpu_topo_test").toFile()
        try {
            // Simulate Snapdragon 7+ Gen 2 (marble): 0..3 @ 1.8 GHz, 4..6 @ 2.5 GHz, 7 @ 2.92 GHz
            val freqs = listOf(1804800, 1804800, 1804800, 1804800, 2496000, 2496000, 2496000, 2918400)
            freqs.forEachIndexed { cpu, khz ->
                val dir = File(tmpRoot, "cpu$cpu/cpufreq").apply { mkdirs() }
                File(dir, "cpuinfo_max_freq").writeText("$khz\n")
            }
            val detected = OrtConfig.detectBigPrimeCpus(maxCount = 4, cpuSysRoot = tmpRoot)
            assertEquals(listOf(7, 6, 5, 4), detected)

            // ORT requires exact (intra_op_num_threads - 1) entries separated by ';'
            val aff4 = OrtConfig.formatIntraOpAffinities(detected, threads = 4)
            assertEquals("4,5,6,7;4,5,6,7;4,5,6,7", aff4)
            assertEquals(3, aff4.split(';').size)

            assertEquals("", OrtConfig.formatIntraOpAffinities(detected, threads = 1))
            assertEquals("", OrtConfig.formatIntraOpAffinities(emptyList(), threads = 4))

            OrtConfig.createSessionOptions(backend = AccelBackend.AUTO, threads = 4, bigPrimeCpus = detected).close()
        } finally {
            tmpRoot.deleteRecursively()
        }
    }

    @Test
    fun cameraSettingsDefaultsToHalAwbMode() {
        assertEquals(AwbEstimatorMode.HAL, CameraSettings().awbMode)
        assertEquals(listOf(AwbEstimatorMode.HAL, AwbEstimatorMode.FFCC, AwbEstimatorMode.C5), AwbEstimatorMode.entries)
    }
}
