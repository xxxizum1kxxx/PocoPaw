package com.izum1k.pocopaw.pipeline

import android.content.Context
import java.io.File
import java.io.InputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Extracts bundled ONNX models from `assets/models/` into `filesDir/models/` so native C++
 * stages (`pmrid.cpp`, `optical_flow.cpp`, `hdrnet.cpp`, `semantic_seg.cpp`) can open them
 * directly by filesystem path.
 */
object ModelCatalog {
    const val ASSET_DIR = "models"
    const val PMRID_FILE = "pmrid.onnx"
    const val SEA_RAFT_FILE = "sea_raft.onnx"
    const val HDRNET_FILE = "hdrnet.onnx"
    const val SEG_FILE = "semantic_seg.onnx"

    private val extractLock = Any()

    fun pmridModelPath(context: Context): String = extractAsset(context, PMRID_FILE)
    fun seaRaftModelPath(context: Context): String = extractAsset(context, SEA_RAFT_FILE)
    fun hdrnetModelPath(context: Context): String = extractAsset(context, HDRNET_FILE)
    fun segModelPath(context: Context): String = extractAsset(context, SEG_FILE)

    fun extractAsset(context: Context, fileName: String): String {
        val filesDir = runCatching { context.filesDir }.getOrNull()
        val assets = runCatching { context.assets }.getOrNull()
        return extractAsset(filesDir, fileName) { path ->
            runCatching { assets?.open(path) }.getOrNull()
        }
    }

    internal fun extractAsset(
        filesDir: File?,
        fileName: String,
        openStream: (String) -> InputStream?,
    ): String {
        if (filesDir == null || fileName.isBlank()) return ""
        if (hasUnsafePathToken(fileName)) return ""
        synchronized(extractLock) {
            val dir = File(filesDir, ASSET_DIR)
            if (!dir.isDirectory) {
                dir.mkdirs()
                if (!dir.isDirectory) return ""
            }
            val dirPath = runCatching { dir.canonicalFile.toPath() }.getOrNull() ?: return ""
            val target = File(dir, fileName)
            val targetPath = runCatching { target.canonicalFile.toPath() }.getOrNull() ?: return ""
            if (targetPath == dirPath || !targetPath.startsWith(dirPath)) return ""
            if (target.isDirectory) return ""
            if (target.isFile && target.length() > 0L) return target.absolutePath
            if (target.isFile && target.length() == 0L) {
                runCatching { target.delete() }
            }

            var tmp: File? = null
            return try {
                val input = runCatching { openStream("$ASSET_DIR/$fileName") }.getOrNull() ?: return ""
                val tmpFile = File.createTempFile(safeTempPrefix(fileName), ".tmp", dir)
                tmp = tmpFile
                var copied = 0L
                input.use { src ->
                    tmpFile.outputStream().use { dst ->
                        copied = src.copyTo(dst)
                    }
                }
                if (copied <= 0L || !tmpFile.isFile || tmpFile.length() == 0L) {
                    if (target.isFile && target.length() == 0L) runCatching { target.delete() }
                    return ""
                }
                if (target.isDirectory) return ""
                try {
                    Files.move(
                        tmpFile.toPath(),
                        target.toPath(),
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING,
                    )
                } catch (_: AtomicMoveNotSupportedException) {
                    Files.move(
                        tmpFile.toPath(),
                        target.toPath(),
                        StandardCopyOption.REPLACE_EXISTING,
                    )
                }
                if (!target.isFile || target.length() == 0L) {
                    runCatching { target.delete() }
                    return ""
                }
                target.absolutePath
            } catch (_: Throwable) {
                if (target.isFile && target.length() == 0L) runCatching { target.delete() }
                ""
            } finally {
                tmp?.let { runCatching { if (it.exists()) it.delete() } }
            }
        }
    }

    private fun hasUnsafePathToken(name: String): Boolean =
        name.contains('/') || name.contains('\\') || name.contains("..")

    private fun safeTempPrefix(fileName: String): String {
        val sanitized = fileName.replace(Regex("[^A-Za-z0-9._-]"), "_").take(40)
        return if (sanitized.length >= 3) sanitized else sanitized.padEnd(3, '_')
    }
}
