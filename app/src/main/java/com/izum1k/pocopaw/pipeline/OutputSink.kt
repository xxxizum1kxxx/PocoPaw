package com.izum1k.pocopaw.pipeline

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

    /** The newest file this app saved, for the thumbnail button after a restart. */
    suspend fun latest(): Uri? = withContext(Dispatchers.IO) {
        runCatching {
            resolver.query(
                collection,
                arrayOf(MediaStore.MediaColumns._ID),
                "${MediaStore.MediaColumns.RELATIVE_PATH} = ?",
                arrayOf(RELATIVE_PATH),
                "${MediaStore.MediaColumns.DATE_ADDED} DESC",
            )?.use { c -> if (c.moveToFirst()) android.content.ContentUris.withAppendedId(collection, c.getLong(0)) else null }
        }.getOrNull()
    }

    suspend fun thumbnail(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        runCatching { resolver.loadThumbnail(uri, Size(THUMB_SIZE, THUMB_SIZE), null) }.getOrNull()
    }

    companion object {
        const val RELATIVE_PATH = "DCIM/PocoPaw/"
        private const val THUMB_SIZE = 256

        fun baseName(takenAtMillis: Long): String =
            "PocoPaw_" + SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(takenAtMillis))
    }
}
