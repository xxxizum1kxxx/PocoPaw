package com.izum1k.pocopaw.feature.p1

import android.content.ContentResolver
import android.net.Uri
import java.io.File
import java.io.IOException
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Zips captures as `<item>/<burst>/<file>`, ready to drop into `pipeline/data/`. Level 1 is
 * enough: raw16 keeps 10-bit samples in 16 bits, so even the fastest deflate saves about a third.
 */
object ZipExporter {
    fun export(resolver: ContentResolver, target: Uri, root: File, items: List<P1Item>, onProgress: (Float) -> Unit) {
        val files = items.flatMap { item -> File(root, item.name).walkTopDown().filter { it.isFile }.toList() }
        val total = files.sumOf { it.length() }.coerceAtLeast(1L)
        var done = 0L
        val out = resolver.openOutputStream(target) ?: throw IOException("Cannot open $target")
        ZipOutputStream(out.buffered(BUFFER)).use { zip ->
            zip.setLevel(Deflater.BEST_SPEED)
            val buf = ByteArray(BUFFER)
            for (f in files) {
                zip.putNextEntry(ZipEntry(f.relativeTo(root).invariantSeparatorsPath).apply { time = f.lastModified() })
                f.inputStream().use { input ->
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        zip.write(buf, 0, n)
                        done += n
                        onProgress(done / total.toFloat())
                    }
                }
                zip.closeEntry()
            }
        }
    }

    private const val BUFFER = 1 shl 20
}
