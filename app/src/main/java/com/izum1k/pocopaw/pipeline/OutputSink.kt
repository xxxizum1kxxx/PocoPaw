package com.izum1k.pocopaw.pipeline

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.util.Size
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

enum class OutputKind(val mimeType: String, val extension: String) {
    DNG("image/x-adobe-dng", "dng"),
    JPEG("image/jpeg", "jpg"),
    ULTRA_HDR("image/jpeg", "jpg"),
}

data class SavedOutput(val kind: OutputKind, val uri: Uri, val displayName: String)

data class LatestOutput(val uri: Uri, val mimeType: String)

internal data class MediaStoreRow(val id: Long, val displayName: String, val mimeType: String)

/** Where a pipeline's files go. Stages only see this, so tests or other targets can replace MediaStore. */
interface OutputSink {
    suspend fun write(kind: OutputKind, baseName: String, takenAtMillis: Long, writer: (OutputStream) -> Unit): SavedOutput
}

/** Writes into `DCIM/PocoPaw` through MediaStore; a file stays pending until it is complete. */
class MediaStoreWriter(context: Context) : OutputSink {
    private val resolver = context.applicationContext.contentResolver
    private val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)

    override suspend fun write(kind: OutputKind, baseName: String, takenAtMillis: Long, writer: (OutputStream) -> Unit): SavedOutput =
        withContext(Dispatchers.IO) {
            val name = "$baseName.${kind.extension}"
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, kind.mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, RELATIVE_PATH)
                put(MediaStore.MediaColumns.DATE_TAKEN, takenAtMillis)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val uri = resolver.insert(collection, values) ?: throw IOException("MediaStore refused $name")
            try {
                resolver.openOutputStream(uri)?.use(writer) ?: throw IOException("No stream for $uri")
                resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            } catch (e: Throwable) {
                resolver.delete(uri, null, null)
                throw e
            }
            SavedOutput(kind, uri, name)
        }

    /** Recent finished captures this app saved, newest first, deduplicating DNG+JPEG pairs. */
    suspend fun recent(limit: Int = MAX_RECENT_REVIEW): List<LatestOutput> = withContext(Dispatchers.IO) {
        if (limit <= 0) return@withContext emptyList()
        runCatching {
            resolver.query(
                collection,
                arrayOf(MediaStore.MediaColumns._ID, MediaStore.MediaColumns.DISPLAY_NAME, MediaStore.MediaColumns.MIME_TYPE),
                "${MediaStore.MediaColumns.RELATIVE_PATH} = ? AND ${MediaStore.MediaColumns.IS_PENDING} = 0",
                arrayOf(RELATIVE_PATH),
                "${MediaStore.MediaColumns.DATE_TAKEN} DESC, ${MediaStore.MediaColumns.MIME_TYPE} ASC, ${MediaStore.MediaColumns._ID} DESC",
            )?.use { c ->
                val rows = sequence {
                    while (c.moveToNext()) {
                        yield(
                            MediaStoreRow(
                                id = c.getLong(0),
                                displayName = c.getString(1).orEmpty(),
                                mimeType = c.getString(2) ?: "image/*",
                            ),
                        )
                    }
                }
                selectRecentPrimaryRows(rows, limit).map { row ->
                    LatestOutput(
                        uri = ContentUris.withAppendedId(collection, row.id),
                        mimeType = row.mimeType,
                    )
                }
            }.orEmpty()
        }.getOrDefault(emptyList())
    }

    /** The newest file this app saved, for the thumbnail button after a restart. */
    suspend fun latest(): LatestOutput? = recent(limit = 1).firstOrNull()

    suspend fun thumbnail(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        runCatching { resolver.loadThumbnail(uri, Size(THUMB_SIZE, THUMB_SIZE), null) }.getOrNull()
    }

    companion object {
        const val RELATIVE_PATH = "DCIM/PocoPaw/"
        const val MAX_RECENT_REVIEW = 64
        private const val THUMB_SIZE = 256

        internal fun selectRecentPrimaryRows(rows: Sequence<MediaStoreRow>, limit: Int): List<MediaStoreRow> {
            if (limit <= 0) return emptyList()
            val byStem = LinkedHashMap<String, MediaStoreRow>(minOf(limit, 64))
            for (row in rows) {
                val stem = row.displayName.substringBeforeLast('.', row.displayName)
                val existing = byStem[stem]
                if (existing == null) {
                    if (byStem.size >= limit) break
                    byStem[stem] = row
                } else if (existing.mimeType == OutputKind.DNG.mimeType && row.mimeType != OutputKind.DNG.mimeType) {
                    byStem[stem] = row
                }
            }
            return byStem.values.toList()
        }

        fun baseName(takenAtMillis: Long): String =
            "PocoPaw_" + SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(takenAtMillis))
    }
}
