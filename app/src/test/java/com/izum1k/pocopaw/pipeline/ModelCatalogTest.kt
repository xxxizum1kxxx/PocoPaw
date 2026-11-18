package com.izum1k.pocopaw.pipeline

import android.content.ContextWrapper
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.io.path.createTempDirectory

class ModelCatalogTest {

    @Test
    fun modelCatalogHandlesEmptyNullCorruptAndMissingAssetStreams() {
        val root = createTempDirectory("model_catalog_test").toFile()
        try {
            // 1. Null filesDir and blank filename fail cleanly without throwing
            assertEquals("", ModelCatalog.extractAsset(null, ModelCatalog.PMRID_FILE) { ByteArrayInputStream(byteArrayOf(1, 2)) })
            assertEquals("", ModelCatalog.extractAsset(root, "  ") { ByteArrayInputStream(byteArrayOf(1, 2)) })

            // 2. JVM Context stub with null filesDir / null assets returns ""
            assertEquals("", ModelCatalog.extractAsset(ContextWrapper(null), ModelCatalog.PMRID_FILE))

            // 3. Null InputStream (e.g. AssetManager stub) returns "" and leaves no file
            assertEquals("", ModelCatalog.extractAsset(root, ModelCatalog.PMRID_FILE) { null })
            assertFalse(File(root, "${ModelCatalog.ASSET_DIR}/${ModelCatalog.PMRID_FILE}").exists())

            // 4. Missing asset (FileNotFoundException) returns "" cleanly
            assertEquals(
                "",
                ModelCatalog.extractAsset(root, ModelCatalog.SEA_RAFT_FILE) {
                    throw FileNotFoundException("missing $it")
                },
            )
            assertFalse(File(root, "${ModelCatalog.ASSET_DIR}/${ModelCatalog.SEA_RAFT_FILE}").exists())

            // 5. Empty 0-byte asset stream returns "", cleans .tmp, and retains no 0-byte target
            assertEquals(
                "",
                ModelCatalog.extractAsset(root, ModelCatalog.HDRNET_FILE) {
                    ByteArrayInputStream(ByteArray(0))
                },
            )
            val hdrnetDir = File(root, ModelCatalog.ASSET_DIR)
            assertFalse(File(hdrnetDir, ModelCatalog.HDRNET_FILE).exists())
            assertFalse(File(hdrnetDir, "${ModelCatalog.HDRNET_FILE}.tmp").exists())

            // 6. Pre-existing 0-byte target is purged and not returned when stream fails mid-read
            val corruptTarget = File(hdrnetDir, ModelCatalog.SEG_FILE).apply {
                parentFile?.mkdirs()
                writeBytes(ByteArray(0))
            }
            val midStreamFail = object : InputStream() {
                private var calls = 0
                override fun read(): Int {
                    if (calls++ < 4) return 0x42
                    throw IOException("corrupt stream mid-read")
                }
            }
            assertEquals("", ModelCatalog.extractAsset(root, ModelCatalog.SEG_FILE) { midStreamFail })
            assertFalse(corruptTarget.exists())
            assertFalse(File(hdrnetDir, "${ModelCatalog.SEG_FILE}.tmp").exists())

            // 7. Valid non-empty stream extracts exact bytes and reuses cached target
            val payload = byteArrayOf(0x08, 0x09, 0x10, 0x20, 0x7F)
            var opens = 0
            val extractedPath = ModelCatalog.extractAsset(root, ModelCatalog.PMRID_FILE) {
                opens++
                ByteArrayInputStream(payload)
            }
            val target = File(extractedPath)
            assertTrue(target.isFile)
            assertArrayEquals(payload, target.readBytes())
            assertEquals(1, opens)

            // Cached non-empty file is reused without re-opening the asset stream
            assertEquals(extractedPath, ModelCatalog.extractAsset(root, ModelCatalog.PMRID_FILE) {
                opens++
                ByteArrayInputStream(payload)
            })
            assertEquals(1, opens)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun lutCatalogHandlesMissingEmptyAndCorruptAssetStreams() {
        val root = createTempDirectory("lut_catalog_test").toFile()
        try {
            assertNull(LutCatalog.pathFor(null, "film_warm") { ByteArrayInputStream(byteArrayOf(1)) })
            assertNull(LutCatalog.pathFor(root, "unknown_lut") { ByteArrayInputStream(byteArrayOf(1)) })
            assertNull(LutCatalog.pathFor(root, "film_warm") { null })
            assertNull(LutCatalog.pathFor(root, "film_warm") { throw FileNotFoundException("missing") })
            assertNull(LutCatalog.pathFor(root, "film_warm") { ByteArrayInputStream(ByteArray(0)) })

            val lutsDir = File(root, LutCatalog.ASSET_DIR)
            assertFalse(File(lutsDir, "film_warm.cube").exists())
            assertFalse(File(lutsDir, "film_warm.cube.tmp").exists())

            val validCube = "LUT_3D_SIZE 2\n0 0 0\n".toByteArray()
            val resolved = LutCatalog.pathFor(root, "film_warm") { ByteArrayInputStream(validCube) }!!
            assertTrue(File(resolved).isFile)
            assertArrayEquals(validCube, File(resolved).readBytes())

            val all = LutCatalog.extractAll(root) { path ->
                if (path.endsWith("bw_silver.cube")) throw IOException("corrupt")
                ByteArrayInputStream(validCube)
            }
            assertEquals(setOf("film_warm", "classic_chrome"), all.keys)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun concurrentModelAndLutExtractionProducesZeroCorruptOrEmptyReads() {
        val modelPayload = ByteArray(512 * 1024) { i -> ((i * 31 + 7) and 0xFF).toByte() }
        val lutPayload = ByteArray(256 * 1024) { i -> ((i * 17 + 3) and 0xFF).toByte() }
        val modelSha = sha256(modelPayload)
        val lutSha = sha256(lutPayload)

        val threads = 16
        val rounds = 20
        val pool = Executors.newFixedThreadPool(threads)
        try {
            // 1. 16-thread x 20-round cold-start race per round on ModelCatalog & LutCatalog
            for (round in 0 until rounds) {
                val root = createTempDirectory("catalog_concurrent_round_$round").toFile()
                try {
                    val gate = CountDownLatch(1)
                    val futures = (0 until threads).map {
                        pool.submit(
                            Callable {
                                gate.await(5, TimeUnit.SECONDS)
                                val mPath = ModelCatalog.extractAsset(root, ModelCatalog.SEA_RAFT_FILE) {
                                    chunkedStream(modelPayload, chunkSize = 4096)
                                }
                                assertTrue("round=$round empty model path", mPath.isNotEmpty())
                                assertEquals("round=$round corrupt model sha", modelSha, sha256(File(mPath).readBytes()))

                                val lPath = LutCatalog.pathFor(root, "film_warm") {
                                    chunkedStream(lutPayload, chunkSize = 4096)
                                }
                                assertTrue("round=$round null lut path", !lPath.isNullOrEmpty())
                                assertEquals("round=$round corrupt lut sha", lutSha, sha256(File(lPath!!).readBytes()))
                            },
                        )
                    }
                    gate.countDown()
                    futures.forEach { it.get(15, TimeUnit.SECONDS) }

                    val leftoverModelTmp = File(root, ModelCatalog.ASSET_DIR).listFiles()?.filter { it.name.endsWith(".tmp") }.orEmpty()
                    val leftoverLutTmp = File(root, LutCatalog.ASSET_DIR).listFiles()?.filter { it.name.endsWith(".tmp") }.orEmpty()
                    assertEquals(emptyList<File>(), leftoverModelTmp)
                    assertEquals(emptyList<File>(), leftoverLutTmp)
                } finally {
                    root.deleteRecursively()
                }
            }

            // 2. 16 threads x 20 rounds hitting a shared root simultaneously
            val sharedRoot = createTempDirectory("catalog_concurrent_shared").toFile()
            try {
                val gate = CountDownLatch(1)
                val futures = (0 until threads).map {
                    pool.submit(
                        Callable {
                            gate.await(5, TimeUnit.SECONDS)
                            repeat(rounds) {
                                val mPath = ModelCatalog.extractAsset(sharedRoot, ModelCatalog.PMRID_FILE) {
                                    chunkedStream(modelPayload, chunkSize = 4096)
                                }
                                assertTrue(mPath.isNotEmpty())
                                assertEquals(modelSha, sha256(File(mPath).readBytes()))

                                val lPath = LutCatalog.pathFor(sharedRoot, "classic_chrome") {
                                    chunkedStream(lutPayload, chunkSize = 4096)
                                }
                                assertTrue(!lPath.isNullOrEmpty())
                                assertEquals(lutSha, sha256(File(lPath!!).readBytes()))
                            }
                        },
                    )
                }
                gate.countDown()
                futures.forEach { it.get(15, TimeUnit.SECONDS) }

                val leftoverModelTmp = File(sharedRoot, ModelCatalog.ASSET_DIR).listFiles()?.filter { it.name.endsWith(".tmp") }.orEmpty()
                val leftoverLutTmp = File(sharedRoot, LutCatalog.ASSET_DIR).listFiles()?.filter { it.name.endsWith(".tmp") }.orEmpty()
                assertEquals(emptyList<File>(), leftoverModelTmp)
                assertEquals(emptyList<File>(), leftoverLutTmp)
            } finally {
                sharedRoot.deleteRecursively()
            }
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun pathTraversalOutsideCatalogDirectoryIsRejected() {
        val sandbox = createTempDirectory("catalog_traversal_test").toFile()
        val filesDir = File(sandbox, "files").apply { mkdirs() }
        try {
            val payload = byteArrayOf(1, 2, 3, 4)
            val maliciousNames = listOf(
                "../escaped.onnx",
                "../../escaped_root.onnx",
                "sub/nested.onnx",
                "..\\escaped_win.onnx",
                "..",
                ".",
            )
            for (name in maliciousNames) {
                assertEquals("Should reject '$name'", "", ModelCatalog.extractAsset(filesDir, name) { ByteArrayInputStream(payload) })
            }
            assertFalse(File(filesDir, "escaped.onnx").exists())
            assertFalse(File(sandbox, "escaped_root.onnx").exists())
            assertNull(LutCatalog.pathFor(filesDir, "../film_warm") { ByteArrayInputStream(payload) })
            assertNull(LutCatalog.pathFor(filesDir, "luts/../film_warm") { ByteArrayInputStream(payload) })
        } finally {
            sandbox.deleteRecursively()
        }
    }

    @Test
    fun directoryTargetIsRejectedWithoutOverwritingEmptyOrNonEmptyDirectory() {
        val root = createTempDirectory("catalog_dir_target_test").toFile()
        try {
            val payload = byteArrayOf(0x0A, 0x0B, 0x0C, 0x0D)

            // Empty directory at model target
            val emptyModelDir = File(root, "${ModelCatalog.ASSET_DIR}/${ModelCatalog.PMRID_FILE}").apply { mkdirs() }
            assertEquals("", ModelCatalog.extractAsset(root, ModelCatalog.PMRID_FILE) { ByteArrayInputStream(payload) })
            assertTrue(emptyModelDir.isDirectory)

            // Non-empty directory at model target
            val nonEmptyModelDir = File(root, "${ModelCatalog.ASSET_DIR}/${ModelCatalog.HDRNET_FILE}").apply { mkdirs() }
            val sentinelModel = File(nonEmptyModelDir, "keep.txt").apply { writeText("keep") }
            assertEquals("", ModelCatalog.extractAsset(root, ModelCatalog.HDRNET_FILE) { ByteArrayInputStream(payload) })
            assertTrue(nonEmptyModelDir.isDirectory)
            assertTrue(sentinelModel.isFile)

            // Empty directory at LUT target
            val emptyLutDir = File(root, "${LutCatalog.ASSET_DIR}/film_warm.cube").apply { mkdirs() }
            assertNull(LutCatalog.pathFor(root, "film_warm") { ByteArrayInputStream(payload) })
            assertTrue(emptyLutDir.isDirectory)
        } finally {
            root.deleteRecursively()
        }
    }

    private fun chunkedStream(bytes: ByteArray, chunkSize: Int): InputStream = object : InputStream() {
        private var offset = 0
        override fun read(): Int = if (offset < bytes.size) bytes[offset++].toInt() and 0xFF else -1
        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (offset >= bytes.size) return -1
            val n = minOf(len, chunkSize, bytes.size - offset)
            System.arraycopy(bytes, offset, b, off, n)
            offset += n
            return n
        }
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
