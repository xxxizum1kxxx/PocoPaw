package com.izum1k.pocopaw.pipeline

import android.content.Context
import java.io.File
import java.io.InputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

data class LutPreset(
    val id: String,
    val fileName: String,
)

/**
 * Resolves bundled `.cube` 3D LUTs from `assets/luts/` into `filesDir/luts/` so native C++ (`finish.cpp`)
 * can load them by filesystem path uniformly with user-imported `.cube` files.
 */
object LutCatalog {
    const val ASSET_DIR = "luts"

    private val extractLock = Any()

    val presets = listOf(
        LutPreset("film_warm", "film_warm.cube"),
        LutPreset("classic_chrome", "classic_chrome.cube"),
        LutPreset("bw_silver", "bw_silver.cube"),
    )

    fun extractAll(context: Context): Map<String, File> {
        val filesDir = runCatching { context.filesDir }.getOrNull()
        val assets = runCatching { context.assets }.getOrNull()
        return extractAll(filesDir) { path -> runCatching { assets?.open(path) }.getOrNull() }
    }

    fun pathFor(context: Context, id: String): String? {
        val filesDir = runCatching { context.filesDir }.getOrNull()
        val assets = runCatching { context.assets }.getOrNull()
        return pathFor(filesDir, id) { path -> runCatching { assets?.open(path) }.getOrNull() }
    }

    internal fun extractAll(
        filesDir: File?,
        openStream: (String) -> InputStream?,
    ): Map<String, File> {
        val result = LinkedHashMap<String, File>(presets.size)
        for (preset in presets) {
            val path = extractPreset(filesDir, preset.fileName, openStream)
            if (path != null) result[preset.id] = File(path)
        }
        return result
    }

    internal fun pathFor(
        filesDir: File?,
        id: String,
        openStream: (String) -> InputStream?,
    ): String? {
        if (hasUnsafePathToken(id)) return null
        val preset = presets.firstOrNull { it.id == id } ?: return null
        return extractPreset(filesDir, preset.fileName, openStream)
    }

    private fun extractPreset(
        filesDir: File?,
        fileName: String,
        openStream: (String) -> InputStream?,
    ): String? {
        if (filesDir == null || fileName.isBlank()) return null
        if (hasUnsafePathToken(fileName)) return null
        synchronized(extractLock) {
            val outDir = File(filesDir, ASSET_DIR)
            if (!outDir.isDirectory) {
                outDir.mkdirs()
                if (!outDir.isDirectory) return null
            }
            val dirPath = runCatching { outDir.canonicalFile.toPath() }.getOrNull() ?: return null
            val target = File(outDir, fileName)
            val targetPath = runCatching { target.canonicalFile.toPath() }.getOrNull() ?: return null
            if (targetPath == dirPath || !targetPath.startsWith(dirPath)) return null
            if (target.isDirectory) return null
            if (target.isFile && target.length() > 0L) return target.absolutePath
            if (target.isFile && target.length() == 0L) runCatching { target.delete() }

            var tmp: File? = null
            return try {
                val input = runCatching { openStream("$ASSET_DIR/$fileName") }.getOrNull() ?: return null
                val tmpFile = File.createTempFile(safeTempPrefix(fileName), ".tmp", outDir)
                tmp = tmpFile
                var copied = 0L
                input.use { src ->
                    tmpFile.outputStream().use { dst ->
                        copied = src.copyTo(dst)
                    }
                }
                if (copied <= 0L || !tmpFile.isFile || tmpFile.length() == 0L) {
                    if (target.isFile && target.length() == 0L) runCatching { target.delete() }
                    return null
                }
                if (target.isDirectory) return null
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
                    return null
                }
                target.absolutePath
            } catch (_: Throwable) {
                if (target.isFile && target.length() == 0L) runCatching { target.delete() }
                null
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
