package com.izum1k.pocopaw.camera

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

class AwbEstimatorTest {

    private val json = Json { ignoreUnknownKeys = true }

    private fun assetFile(relPath: String): File {
        val candidates = listOf(
            File("src/main/assets/$relPath"),
            File("app/src/main/assets/$relPath"),
        )
        return candidates.firstOrNull { it.isFile }
            ?: error("Missing asset file $relPath (cwd=${File(".").absolutePath})")
    }

    private fun loadResourceBytes(resPath: String): ByteArray =
        requireNotNull(javaClass.classLoader?.getResourceAsStream(resPath)) {
            "Missing test resource: $resPath"
        }.use { it.readBytes() }

    private fun loadFloat32Rgb(resPath: String): FloatArray {
        val bytes = loadResourceBytes(resPath)
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    private fun loadReference(): JsonObject {
        val text = loadResourceBytes("awb/awb_reference.json").decodeToString()
        return json.parseToJsonElement(text).jsonObject
    }

    private fun JsonObject.vec3(key: String): FloatArray {
        val arr = getValue(key).jsonArray
        return FloatArray(3) { arr[it].jsonPrimitive.double.toFloat() }
    }

    private fun angleDeg(a: FloatArray, b: FloatArray): Double {
        val ax = a[0].toDouble(); val ay = a[1].toDouble(); val az = a[2].toDouble()
        val bx = b[0].toDouble(); val by = b[1].toDouble(); val bz = b[2].toDouble()
        val cx = ay * bz - az * by
        val cy = az * bx - ax * bz
        val cz = ax * by - ay * bx
        val crossNorm = sqrt(cx * cx + cy * cy + cz * cz)
        val dot = ax * bx + ay * by + az * bz
        return Math.toDegrees(atan2(crossNorm, dot))
    }

    private fun maxAbs(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (i in 0 until 3) m = max(m, abs(a[i] - b[i]))
        return m
    }

    private fun dummyMeterStats(camRgb: FloatArray, valid: BooleanArray): MeterStats =
        MeterStats(
            timestampNs = 1L,
            sceneEv = 0.0,
            grid = FloatArray(SceneMeter.GRID_COLS * SceneMeter.GRID_ROWS),
            camRgb = camRgb,
            valid = valid,
            gains = floatArrayOf(1f, 1f, 1f),
            tonemapFromResult = true,
            sampleMillis = 0.1,
            millis = 0.1,
        )

    @Test
    fun halEstimatorReturnsNull() {
        val rgb = FloatArray(SceneMeter.COLS * SceneMeter.ROWS * 3) { 0.5f }
        val valid = BooleanArray(SceneMeter.COLS * SceneMeter.ROWS) { true }
        assertNull(AwbEstimator.HAL.estimate(dummyMeterStats(rgb, valid), "0"))
    }

    @Test
    fun fft2dRoundTripMatchesIdentity() {
        val re = DoubleArray(Fft2d64.SIZE) { i -> ((i * 37) % 101) / 100.0 - 0.5 }
        val im = DoubleArray(Fft2d64.SIZE) { i -> ((i * 19) % 53) / 50.0 - 0.5 }
        val origRe = re.copyOf()
        val origIm = im.copyOf()
        Fft2d64.fft2(re, im)
        Fft2d64.ifft2(re, im)
        var maxErr = 0.0
        for (i in 0 until Fft2d64.SIZE) {
            maxErr = max(maxErr, abs(re[i] - origRe[i]))
            maxErr = max(maxErr, abs(im[i] - origIm[i]))
        }
        assertTrue("FFT roundtrip error too large: $maxErr", maxErr < 1e-11)
    }

    @Test
    fun ffccMatchesPythonReferenceAcrossResolutionsAndFallback() {
        val est = FfccAwbEstimator.fromFiles(
            mapOf(
                "0" to assetFile("calib/ffcc_cam0.bin"),
                "2" to assetFile("calib/ffcc_cam2.bin"),
            ),
            usePrior = false,
        )
        val refRoot = loadReference()
        for (scene in listOf("awb_warm", "awb_cool")) {
            val sceneObj = refRoot.getValue(scene).jsonObject
            val gt = sceneObj.vec3("gt")
            for ((res, dims) in listOf("384x256" to (384 to 256), "64x48" to (64 to 48))) {
                val (w, h) = dims
                val rgb = loadFloat32Rgb("awb/${scene}_$res.f32")
                val resObj = sceneObj.getValue(res).jsonObject

                val priorEst = est.estimateIlluminant(rgb, null, w, h, "0", usePrior = true)
                assertNotNull("FFCC prior failed on $scene $res", priorEst)
                val priorRef = resObj.vec3("ffcc_prior")
                assertTrue(
                    "FFCC prior $scene $res angle=${angleDeg(priorEst!!, priorRef)} est=${priorEst.contentToString()} ref=${priorRef.contentToString()}",
                    angleDeg(priorEst, priorRef) <= 0.01,
                )
                assertTrue(
                    "FFCC prior $scene $res maxAbs=${maxAbs(priorEst, priorRef)}",
                    maxAbs(priorEst, priorRef) <= 1e-4f,
                )

                val nopriorEst = est.estimateIlluminant(rgb, null, w, h, "0", usePrior = false)
                assertNotNull("FFCC noprior failed on $scene $res", nopriorEst)
                val nopriorRef = resObj.vec3("ffcc_noprior")
                assertTrue(
                    "FFCC noprior $scene $res angle=${angleDeg(nopriorEst!!, nopriorRef)}",
                    angleDeg(nopriorEst, nopriorRef) <= 0.01,
                )
                assertTrue(
                    "FFCC noprior $scene $res maxAbs=${maxAbs(nopriorEst, nopriorRef)}",
                    maxAbs(nopriorEst, nopriorRef) <= 1e-4f,
                )

                if (res == "384x256") {
                    val degPrior = angleDeg(priorEst, gt)
                    val degNoPrior = angleDeg(nopriorEst, gt)
                    if (scene == "awb_warm") {
                        assertEquals(2.45, degPrior, 0.05)
                        assertEquals(1.92, degNoPrior, 0.05)
                    } else {
                        assertEquals(0.70, degPrior, 0.05)
                        assertEquals(5.50, degNoPrior, 0.05)
                    }
                }
            }
        }

        // Fallback cases: unknown camera ("1"), missing camera, all-invalid mask, all-black input
        val warm64 = loadFloat32Rgb("awb/awb_warm_64x48.f32")
        assertNull(est.estimateIlluminant(warm64, null, 64, 48, "1"))
        assertNull(est.estimateIlluminant(warm64, null, 64, 48, null))
        assertNull(est.estimateIlluminant(warm64, BooleanArray(64 * 48) { false }, 64, 48, "0"))
        assertNull(est.estimateIlluminant(FloatArray(64 * 48 * 3), null, 64, 48, "0"))
        assertNotNull(est.estimateIlluminant(warm64, null, 64, 48, "2"))
    }

    @Test
    fun c5OnnxMatchesPythonReferenceAndHandlesFallback() {
        C5AwbEstimator.fromFile(assetFile("models/c5_m7.onnx")).use { c5 ->
            val refRoot = loadReference()
            val warm384 = loadFloat32Rgb("awb/awb_warm_384x256.f32")
            val cool384 = loadFloat32Rgb("awb/awb_cool_384x256.f32")
            val warm64 = loadFloat32Rgb("awb/awb_warm_64x48.f32")
            val cool64 = loadFloat32Rgb("awb/awb_cool_64x48.f32")

            val hWarm384 = c5.computeHistogram(warm384, null, 384, 256)!!
            val hCool384 = c5.computeHistogram(cool384, null, 384, 256)!!
            val hWarm64 = c5.computeHistogram(warm64, null, 64, 48)!!
            val hCool64 = c5.computeHistogram(cool64, null, 64, 48)!!

            val cases = listOf(
                Triple("awb_warm", "384x256", Pair(warm384 to (384 to 256), listOf(hCool384))),
                Triple("awb_cool", "384x256", Pair(cool384 to (384 to 256), listOf(hWarm384))),
                Triple("awb_warm", "64x48", Pair(warm64 to (64 to 48), listOf(hCool64))),
                Triple("awb_cool", "64x48", Pair(cool64 to (64 to 48), listOf(hWarm64))),
            )

            for ((scene, res, payload) in cases) {
                val (imgPair, pairedExtras) = payload
                val (rgb, dims) = imgPair
                val (w, h) = dims
                val sceneObj = refRoot.getValue(scene).jsonObject
                val gt = sceneObj.vec3("gt")
                val resObj = sceneObj.getValue(res).jsonObject

                val solo = c5.estimateIlluminant(rgb, null, w, h, "0", extras = emptyList())
                assertNotNull("C5 solo failed on $scene $res", solo)
                val soloRef = resObj.vec3("c5_solo")
                assertTrue(
                    "C5 solo $scene $res angle=${angleDeg(solo!!, soloRef)}",
                    angleDeg(solo, soloRef) <= 0.01,
                )
                assertTrue(
                    "C5 solo $scene $res maxAbs=${maxAbs(solo, soloRef)}",
                    maxAbs(solo, soloRef) <= 1e-4f,
                )

                val paired = c5.estimateIlluminant(rgb, null, w, h, "0", extras = pairedExtras)
                assertNotNull("C5 paired failed on $scene $res", paired)
                val pairedRef = resObj.vec3("c5_paired")
                assertTrue(
                    "C5 paired $scene $res angle=${angleDeg(paired!!, pairedRef)}",
                    angleDeg(paired, pairedRef) <= 0.01,
                )
                assertTrue(
                    "C5 paired $scene $res maxAbs=${maxAbs(paired, pairedRef)}",
                    maxAbs(paired, pairedRef) <= 1e-4f,
                )

                if (res == "384x256") {
                    val deg = angleDeg(paired, gt)
                    if (scene == "awb_warm") assertEquals(2.89, deg, 0.05)
                    else assertEquals(0.56, deg, 0.05)
                }
            }

            // Stateful extras buffer mode (extras = null) and safe invalid/black fallbacks
            assertNotNull(c5.estimateIlluminant(warm64, null, 64, 48, "0"))
            assertNotNull(c5.estimateIlluminant(cool64, null, 64, 48, "0"))
            assertNull(c5.estimateIlluminant(warm64, BooleanArray(64 * 48) { false }, 64, 48, "0"))
            assertNull(c5.estimateIlluminant(FloatArray(64 * 48 * 3), null, 64, 48, "0"))
        }
    }
}
