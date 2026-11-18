package com.izum1k.pocopaw.feature.camera

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ContentUris
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal const val ACTION_CAMERA_REVIEW_LEGACY = "com.android.camera.action.REVIEW"
internal const val IMAGE_WILDCARD = "image/*"
internal const val IMAGE_CURSOR_DIR = "vnd.android.cursor.dir/image"

private const val TAG = "PocoPawGallery"

internal data class MediaStoreItem(val id: Long, val displayName: String, val mimeType: String)

internal data class GalleryLaunchSpec<U>(
    val action: String,
    val data: U? = null,
    val mimeType: String? = null,
    val clipItems: List<U> = emptyList(),
    val grantReadUri: Boolean = false,
    val categorySelector: String? = null,
    val useExternalImagesDir: Boolean = false,
    val newTask: Boolean = false,
)

internal fun <U> buildGalleryLaunchSpecs(
    primaryUri: U?,
    mimeType: String?,
    recentUris: List<U>,
): List<GalleryLaunchSpec<U>> {
    if (primaryUri == null) {
        return listOf(
            GalleryLaunchSpec(
                action = Intent.ACTION_MAIN,
                categorySelector = Intent.CATEGORY_APP_GALLERY,
                newTask = true,
            ),
            GalleryLaunchSpec(
                action = Intent.ACTION_VIEW,
                mimeType = IMAGE_CURSOR_DIR,
                useExternalImagesDir = true,
                newTask = true,
            ),
            GalleryLaunchSpec(
                action = Intent.ACTION_VIEW,
                mimeType = IMAGE_WILDCARD,
                newTask = true,
            ),
        )
    }

    val cleanMime = mimeType?.takeIf { it.isNotBlank() } ?: IMAGE_WILDCARD
    val secondary = recentUris.asSequence().filter { it != primaryUri }.distinct().toList()

    return buildList {
        add(GalleryLaunchSpec(MediaStore.ACTION_REVIEW, primaryUri, cleanMime, clipItems = secondary, grantReadUri = true))
        if (cleanMime != IMAGE_WILDCARD) {
            add(GalleryLaunchSpec(MediaStore.ACTION_REVIEW, primaryUri, IMAGE_WILDCARD, clipItems = secondary, grantReadUri = true))
        }
        add(GalleryLaunchSpec(ACTION_CAMERA_REVIEW_LEGACY, primaryUri, cleanMime, clipItems = secondary, grantReadUri = true))
        if (cleanMime != IMAGE_WILDCARD) {
            add(GalleryLaunchSpec(ACTION_CAMERA_REVIEW_LEGACY, primaryUri, IMAGE_WILDCARD, clipItems = secondary, grantReadUri = true))
        }
        add(GalleryLaunchSpec(Intent.ACTION_VIEW, primaryUri, cleanMime, grantReadUri = true))
        if (cleanMime != IMAGE_WILDCARD) {
            add(GalleryLaunchSpec(Intent.ACTION_VIEW, primaryUri, IMAGE_WILDCARD, grantReadUri = true))
        }
    }
}

/** Deduplicates paired captures by stem, preferring JPEG over DNG, up to [limit] shots. */
internal fun selectRecentPrimaryItems(rows: Sequence<MediaStoreItem>, limit: Int = 64): List<MediaStoreItem> {
    if (limit <= 0) return emptyList()
    val byStem = LinkedHashMap<String, MediaStoreItem>()
    for (row in rows) {
        val stem = row.displayName.substringBeforeLast('.')
        val existing = byStem[stem]
        if (existing == null) {
            if (byStem.size >= limit) break
            byStem[stem] = row
        } else if (existing.mimeType == "image/x-adobe-dng" && row.mimeType != "image/x-adobe-dng") {
            byStem[stem] = row
        }
    }
    return byStem.values.toList()
}

suspend fun queryAppRecentCaptures(context: Context, limit: Int = 64): List<Pair<Uri, String>> = withContext(Dispatchers.IO) {
    runCatching {
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        context.applicationContext.contentResolver.query(
            collection,
            arrayOf(
                MediaStore.MediaColumns._ID,
                MediaStore.MediaColumns.DISPLAY_NAME,
                MediaStore.MediaColumns.MIME_TYPE,
            ),
            "${MediaStore.MediaColumns.OWNER_PACKAGE_NAME} = ?",
            arrayOf(context.packageName),
            "${MediaStore.MediaColumns.DATE_TAKEN} DESC, ${MediaStore.MediaColumns._ID} DESC",
        )?.use { cursor ->
            val rows = sequence {
                while (cursor.moveToNext()) {
                    val id = cursor.getLong(0)
                    val name = cursor.getString(1) ?: id.toString()
                    val mime = cursor.getString(2) ?: IMAGE_WILDCARD
                    yield(MediaStoreItem(id, name, mime))
                }
            }
            selectRecentPrimaryItems(rows, limit).map { item ->
                ContentUris.withAppendedId(collection, item.id) to item.mimeType
            }
        }.orEmpty()
    }.getOrDefault(emptyList())
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}

private fun GalleryLaunchSpec<Uri>.toIntent(hasActivity: Boolean): Intent {
    val intent = if (categorySelector != null) {
        Intent.makeMainSelectorActivity(action, categorySelector)
    } else {
        Intent(action)
    }

    val targetUri = data ?: if (useExternalImagesDir) MediaStore.Images.Media.EXTERNAL_CONTENT_URI else null
    if (targetUri != null && mimeType != null) {
        intent.setDataAndType(targetUri, mimeType)
    } else if (targetUri != null) {
        intent.data = targetUri
    } else if (mimeType != null) {
        intent.type = mimeType
    }

    if (clipItems.isNotEmpty()) {
        val clip = ClipData(null, arrayOf(IMAGE_WILDCARD), ClipData.Item(clipItems.first()))
        for (i in 1 until clipItems.size) {
            clip.addItem(ClipData.Item(clipItems[i]))
        }
        intent.clipData = clip
    }

    if (grantReadUri) intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    if (newTask || !hasActivity) intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    return intent
}

fun openGallery(context: Context, last: LastCapture?, recentUris: List<Uri> = emptyList()): Boolean {
    val activity = context.findActivity()
    val launcher = activity ?: context
    val specs = buildGalleryLaunchSpecs(last?.uri, last?.mimeType, recentUris)
    for (spec in specs) {
        val intent = spec.toIntent(hasActivity = activity != null)
        try {
            launcher.startActivity(intent)
            return true
        } catch (_: ActivityNotFoundException) {
            // Try the next fallback.
        } catch (e: SecurityException) {
            Log.w(TAG, "Gallery intent rejected (${spec.action})", e)
        } catch (e: Exception) {
            Log.w(TAG, "Gallery intent failed (${spec.action})", e)
        }
    }
    return false
}
