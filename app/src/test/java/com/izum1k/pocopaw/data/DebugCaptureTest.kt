package com.izum1k.pocopaw.data

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import com.izum1k.pocopaw.camera.CfaPattern
import com.izum1k.pocopaw.camera.ExposurePlan
import com.izum1k.pocopaw.camera.LensFacing
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import com.izum1k.pocopaw.pipeline.BurstStream
import com.izum1k.pocopaw.pipeline.ImagePipeline
import com.izum1k.pocopaw.pipeline.OutputKind
import com.izum1k.pocopaw.pipeline.OutputSink
import com.izum1k.pocopaw.pipeline.PipelineResult
import com.izum1k.pocopaw.pipeline.RawBurst
import com.izum1k.pocopaw.pipeline.RawFrame
import com.izum1k.pocopaw.pipeline.RawMetadata
import com.izum1k.pocopaw.pipeline.SavedOutput
import com.izum1k.pocopaw.pipeline.StageTiming
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DebugCaptureTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun writeRawBufferWritesLittleEndianSamplesWithoutMutatingPosition() {
        val dir = tempFolder.newFolder("raw_test")
        val target = File(dir, DebugCapture.rawFileName(0))
        val samples = shortArrayOf(64, 128, 511, 1023, 4095, 16383)
        val buf = ByteBuffer.allocateDirect(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        samples.forEach(buf::putShort)
        buf.position(4) // simulate non-zero position before duplicate+rewind

        val written = DebugCapture.writeRawBuffer(target, buf)

        assertEquals(samples.size * 2, written)
        assertEquals(4, buf.position())
        val readBack = ByteBuffer.wrap(target.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val actual = ShortArray(samples.size) { readBack.short }
        assertArrayEquals(samples, actual)
    }

    @Test
    fun computeGainsSplitsAnalogAndSensorPlusPostRawDigitalGain() {
        // Within analog range: ISO 400 with min 50, maxAnalog 800, boost 100 -> 8x analog, 1x digital
        val g1 = DebugCapture.computeGains(iso = 400, minIso = 50, maxAnalogIso = 800, postRawBoost = 100)
        assertEquals(8.0, g1.analogGain!!, 1e-6)
        assertEquals(1.0, g1.sensorDigitalGain, 1e-6)
        assertEquals(1.0, g1.postRawDigitalGain, 1e-6)
        assertEquals(1.0, g1.totalDigitalGain, 1e-6)
        assertEquals(8.0, g1.totalGain!!, 1e-6)

        // Above analog range + post-RAW boost: ISO 3200 with min 50, maxAnalog 800, boost 150
        // -> analog = 800/50 = 16x, sensorDigital = 3200/800 = 4x, postRaw = 1.5x, totalDigital = 6x, total = 96x
        val g2 = DebugCapture.computeGains(iso = 3200, minIso = 50, maxAnalogIso = 800, postRawBoost = 150)
        assertEquals(16.0, g2.analogGain!!, 1e-6)
        assertEquals(4.0, g2.sensorDigitalGain, 1e-6)
        assertEquals(1.5, g2.postRawDigitalGain, 1e-6)
        assertEquals(6.0, g2.totalDigitalGain, 1e-6)
        assertEquals(96.0, g2.totalGain!!, 1e-6)
    }

    @Test
    fun createShotDirAndWriteManifestRoundTripCompleteMetadata() {
        val root = tempFolder.newFolder("debug_captures")
        val t0 = 1_763_510_400_123L
        val shotDir1 = DebugCapture.createShotDir(root, t0, "2")
        val shotDir2 = DebugCapture.createShotDir(root, t0, "2")
        assertTrue(shotDir1.isDirectory)
        assertTrue(shotDir2.isDirectory)
        assertFalse(shotDir1.name == shotDir2.name)

        val raw0 = ByteBuffer.allocateDirect(16).order(ByteOrder.LITTLE_ENDIAN).apply {
            repeat(8) { putShort((100 + it).toShort()) }
            flip()
        }
        val raw1 = ByteBuffer.allocateDirect(16).order(ByteOrder.LITTLE_ENDIAN).apply {
            repeat(8) { putShort((200 + it).toShort()) }
            flip()
        }
        DebugCapture.writeRawBuffer(File(shotDir1, DebugCapture.rawFileName(0)), raw0)
        DebugCapture.writeRawBuffer(File(shotDir1, DebugCapture.rawFileName(1)), raw1)

        val plan = ExposurePlan.PerFrame(
            listOf(ManualExposure(100, 8_000_000L), ManualExposure(200, 16_000_000L)),
        )
        val live = LiveExposure(
            iso = 125,
            exposureNs = 10_000_000L,
            focusLocked = true,
            aeConverged = true,
            focusDioptres = 1.25f,
            motionRadPerSec = Float.NaN, // NaN must sanitize to null for strict JSON
        )
        val manifest = CaptureDebugManifest(
            shotId = shotDir1.name,
            captureTimeMillis = t0,
            captureTimeIsoUtc = DebugCapture.formatIsoUtc(t0),
            packageName = "com.izum1k.pocopaw.opt",
            appVersion = "0.5.0",
            deviceManufacturer = "Xiaomi",
            deviceModel = "23049PCD8G",
            deviceBoard = "marble",
            cameraId = "2",
            facing = "BACK",
            cfaPattern = "BGGR",
            cfaArrangement = 3,
            orientationDegrees = 90,
            sensorOrientation = 90,
            zoomRatio = 1f,
            cropZoom = 1f,
            aspectRatio = "4:3",
            activeArray = RectJson(0, 0, 4000, 3000),
            staticBlackLevel = listOf(64, 64, 64, 64),
            staticWhiteLevel = 1023,
            maxAnalogSensitivity = 800,
            sensitivityRange = IntRangeJson(50, 6400),
            exposureTimeRangeNs = LongRangeJson(20_000L, 32_000_000_000L),
            exposurePlan = DebugCapture.serializeExposurePlan(plan),
            previewMetering = DebugCapture.serializePreviewMetering(live),
            pipelineId = "indigo_hdrplus",
            frameCount = 2,
            referenceFrameIndex = 1,
            totalPipelineMs = 312L,
            stageTimings = listOf(
                StageTimingJson("reference", 2L),
                StageTimingJson("align", 85L),
                StageTimingJson("merge", 95L),
                StageTimingJson("finish", 110L),
                StageTimingJson("jpeg", 20L),
            ),
            savedOutputs = listOf(
                SavedOutputJson("JPEG", "PocoPawOpt_20261119.jpg", "content://media/external/images/media/42"),
            ),
            frames = listOf(
                FrameDebugInfo(
                    index = 0,
                    isReference = false,
                    file = "frame_00.raw",
                    width = 4,
                    height = 2,
                    rowStride = 8,
                    byteLength = 16,
                    sensorTimestampNs = 100_000_000L,
                    physicalCameraId = "2",
                    exposureTimeNs = 8_000_000L,
                    sensitivityIso = 100,
                    analogGain = 2.0,
                    digitalGain = 1.0,
                    totalGain = 2.0,
                    blackLevel = listOf(64f, 64f, 64f, 64f),
                    whiteLevel = 1023,
                    awbNeutral = listOf(0.52, 1.0, 0.61),
                    awbGainsRggb = listOf(1.92f, 1.0f, 1.0f, 1.64f),
                    ccm = listOf(1.5, -0.4, -0.1, -0.2, 1.3, -0.1, 0.0, -0.3, 1.3),
                    lensShadingMap = LensShadingJson(rows = 2, cols = 2, gainsRggb = List(16) { 1.1f }),
                    cropRegion = RectJson(0, 0, 4000, 3000),
                ),
                FrameDebugInfo(
                    index = 1,
                    isReference = true,
                    file = "frame_01.raw",
                    width = 4,
                    height = 2,
                    rowStride = 8,
                    byteLength = 16,
                    sensorTimestampNs = 133_000_000L,
                    physicalCameraId = "2",
                    exposureTimeNs = 16_000_000L,
                    sensitivityIso = 200,
                    analogGain = 4.0,
                    digitalGain = 1.0,
                    totalGain = 4.0,
                    blackLevel = listOf(64f, 64f, 64f, 64f),
                    whiteLevel = 1023,
                    awbNeutral = listOf(0.52, 1.0, 0.61),
                    awbGainsRggb = listOf(1.92f, 1.0f, 1.0f, 1.64f),
                    ccm = listOf(1.5, -0.4, -0.1, -0.2, 1.3, -0.1, 0.0, -0.3, 1.3),
                    lensShadingMap = LensShadingJson(rows = 2, cols = 2, gainsRggb = List(16) { 1.1f }),
                    cropRegion = RectJson(0, 0, 4000, 3000),
                ),
            ),
        )

        val jsonFile = DebugCapture.writeManifest(shotDir1, manifest)
        assertTrue(jsonFile.isFile)
        assertFalse(File(shotDir1, "${DebugCapture.MANIFEST_FILE}.tmp").exists())

        val decoded = Json.decodeFromString<CaptureDebugManifest>(jsonFile.readText(Charsets.UTF_8))
        assertEquals("com.izum1k.pocopaw.opt", decoded.packageName)
        assertEquals(1, decoded.referenceFrameIndex)
        assertEquals("PerFrame", decoded.exposurePlan.mode)
        assertEquals(2, decoded.exposurePlan.perFrame?.size)
        assertNull(decoded.previewMetering.motionRadPerSec)
        assertEquals(5, decoded.stageTimings.size)
        assertEquals(312L, decoded.totalPipelineMs)
        assertFalse(decoded.frames[0].isReference)
        assertTrue(decoded.frames[1].isReference)
        assertEquals(16, File(shotDir1, decoded.frames[0].file).length())
        assertEquals(16, File(shotDir1, decoded.frames[1].file).length())
    }

    @Test
    fun debugCapturePipelineDisablesStreamingDumpsFramesAndRecordsTimings() = runBlocking {
        val root = tempFolder.newFolder("pipeline_debug")
        val f0Buf = ByteBuffer.allocateDirect(8).order(ByteOrder.LITTLE_ENDIAN).apply {
            putShort(10); putShort(20); putShort(30); putShort(40); flip()
        }
        val f1Buf = ByteBuffer.allocateDirect(8).order(ByteOrder.LITTLE_ENDIAN).apply {
            putShort(50); putShort(60); putShort(70); putShort(80); flip()
        }
        val stubResult = allocateStub(CaptureResult::class.java)
        val stubChars = allocateStub(CameraCharacteristics::class.java)
        val burst = RawBurst(
            frames = listOf(
                RawFrame(f0Buf, width = 2, height = 2, timestampNs = 1_000_000L, result = stubResult),
                RawFrame(f1Buf, width = 2, height = 2, timestampNs = 2_000_000L, result = stubResult),
            ),
            metadata = RawMetadata(
                cameraId = "2",
                facing = LensFacing.BACK,
                characteristics = stubChars,
                cfa = CfaPattern.BGGR,
                blackLevel = floatArrayOf(64f, 64f, 64f, 64f),
                whiteLevel = 1023,
                wbGains = null,
                colorTransform = null,
                neutralPoint = null,
                lensShading = null,
                hasNoiseProfile = false,
                orientationDegrees = 90,
                zoomRatio = 1f,
                cropZoom = 1f,
                captureTimeMillis = 1_763_510_500_000L,
            ),
        )
        val fakeDelegate = object : ImagePipeline {
            override val id: String = "indigo_hdrplus"
            override val framesPerCapture: Int = 8
            override fun openStream(exposure: ExposurePlan): BurstStream? = error("must not be called when wrapped")
            override suspend fun process(burst: RawBurst, sink: OutputSink, progress: (Float) -> Unit): PipelineResult {
                burst.referenceIndex = 1
                progress(0.5f)
                progress(1f)
                return PipelineResult(
                    outputs = emptyList(),
                    thumbnail = null,
                    timings = listOf(StageTiming("align", 42L), StageTiming("merge", 58L), StageTiming("finish", 70L)),
                )
            }
        }
        val fakeSink = object : OutputSink {
            override suspend fun write(
                kind: OutputKind,
                baseName: String,
                takenAtMillis: Long,
                writer: (java.io.OutputStream) -> Unit,
            ): SavedOutput = error("unused")
        }
        val wrapped = DebugCapturePipeline(
            delegate = fakeDelegate,
            debugRoot = root,
            packageName = "com.izum1k.pocopaw.opt",
            versionName = "0.5.0",
            aspect = AspectRatio.SIXTEEN_NINE,
            exposurePlan = ExposurePlan.Manual(ManualExposure(100, 12_000_000L)),
            liveExposure = LiveExposure(iso = 200, exposureNs = 12_000_000L),
        )

        assertNull(wrapped.openStream(ExposurePlan.Auto()))
        assertEquals(8, wrapped.framesPerCapture)

        val progresses = mutableListOf<Float>()
        val res = wrapped.process(burst, fakeSink) { progresses += it }
        assertEquals(170L, res.totalMillis)
        assertTrue(progresses.last() == 1f)

        val shotDirs = root.listFiles()?.filter { it.isDirectory }.orEmpty()
        assertEquals(1, shotDirs.size)
        val dir = shotDirs.single()
        assertTrue(File(dir, "frame_00.raw").isFile)
        assertTrue(File(dir, "frame_01.raw").isFile)
        val manifest = Json.decodeFromString<CaptureDebugManifest>(File(dir, DebugCapture.MANIFEST_FILE).readText())
        assertEquals(1, manifest.referenceFrameIndex)
        assertEquals("16:9", manifest.aspectRatio)
        assertEquals("Manual", manifest.exposurePlan.mode)
        assertEquals(170L, manifest.totalPipelineMs)
        assertEquals(listOf("align", "merge", "finish"), manifest.stageTimings.map { it.stage })
        assertFalse(manifest.frames[0].isReference)
        assertTrue(manifest.frames[1].isReference)
        assertNull(manifest.pipelineError)
    }

    private fun <T> allocateStub(cls: Class<T>): T {
        val unsafeClass = Class.forName("sun.misc.Unsafe")
        val theUnsafe = unsafeClass.getDeclaredField("theUnsafe").apply { isAccessible = true }.get(null)
        @Suppress("UNCHECKED_CAST")
        return unsafeClass.getMethod("allocateInstance", Class::class.java).invoke(theUnsafe, cls) as T
    }
}
