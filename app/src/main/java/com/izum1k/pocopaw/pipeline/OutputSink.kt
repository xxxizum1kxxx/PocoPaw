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

data class LatestOutput(val uri: Uri, val mimeType: String)

/** Where a pipeline's files go. Stages only see this, so tests or other targets can replace MediaStore. */
interface OutputSink {
    suspend fun write(kind: OutputKind, baseName: String, takenAtMillis: Long, writer: (OutputStream) -> Unit): SavedOutput
}

/** Writes into `DCIM/PocoPaw` (or `DCIM/PocoPawOpt` for the side-by-side `.opt` package) through MediaStore. */
class MediaStoreWriter(context: Context) : OutputSink {
    private val appContext = context.applicationContext
    private val resolver = appContext.contentResolver
    private val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
    private val isOpt = appContext.packageName.endsWith(".opt")
    private val relativePath = if (isOpt) RELATIVE_PATH_OPT else RELATIVE_PATH

    override suspend fun write(kind: OutputKind, baseName: String, takenAtMillis: Long, writer: (OutputStream) -> Unit): SavedOutput =
        withContext(Dispatchers.IO) {
            val resolvedBase = if (isOpt && baseName.startsWith("PocoPaw_")) {
                "PocoPawOpt_" + baseName.removePrefix("PocoPaw_")
            } else {
                baseName
            }
            val name = "$resolvedBase.${kind.extension}"
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, kind.mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, relativePath)
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
    suspend fun latest(): LatestOutput? = withContext(Dispatchers.IO) {
        runCatching {
            resolver.query(
                collection,
                arrayOf(MediaStore.MediaColumns._ID, MediaStore.MediaColumns.MIME_TYPE),
                "${MediaStore.MediaColumns.RELATIVE_PATH} = ?",
                arrayOf(relativePath),
                "${MediaStore.MediaColumns.DATE_TAKEN} DESC, ${MediaStore.MediaColumns.MIME_TYPE} ASC, ${MediaStore.MediaColumns._ID} DESC",
            )?.use { c ->
                if (!c.moveToFirst()) null
                else LatestOutput(
                    uri = android.content.ContentUris.withAppendedId(collection, c.getLong(0)),
                    mimeType = c.getString(1) ?: "image/*",
                )
            }
        }.getOrNull()
    }

    suspend fun thumbnail(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        runCatching { resolver.loadThumbnail(uri, Size(THUMB_SIZE, THUMB_SIZE), null) }.getOrNull()
    }

    companion object {
        const val RELATIVE_PATH = "DCIM/PocoPaw/"
        const val RELATIVE_PATH_OPT = "DCIM/PocoPawOpt/"
        private const val THUMB_SIZE = 256

        fun baseName(takenAtMillis: Long): String =
            "PocoPaw_" + SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(takenAtMillis))
    }
}
