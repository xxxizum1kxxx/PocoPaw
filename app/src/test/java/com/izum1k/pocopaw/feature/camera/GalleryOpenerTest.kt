package com.izum1k.pocopaw.feature.camera

import android.content.Intent
import android.provider.MediaStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GalleryOpenerTest {

    @Test
    fun captureSpecsPreferReviewIntentWithClipItemsAndFallBackToView() {
        val primary = "content://media/external/images/media/101"
        val recents = listOf(
            primary,
            "content://media/external/images/media/100",
            "content://media/external/images/media/99",
        )
        val specs = buildGalleryLaunchSpecs(
            primaryUri = primary,
            mimeType = "image/jpeg",
            recentUris = recents,
        )

        assertEquals(6, specs.size)
        assertEquals(MediaStore.ACTION_REVIEW, specs[0].action)
        assertEquals(primary, specs[0].data)
        assertEquals("image/jpeg", specs[0].mimeType)
        assertEquals(
            listOf(
                "content://media/external/images/media/100",
                "content://media/external/images/media/99",
            ),
            specs[0].clipItems,
        )
        assertTrue(specs[0].grantReadUri)
        assertEquals(Intent.ACTION_VIEW, specs[4].action)
        assertTrue(specs[4].clipItems.isEmpty())
    }

    @Test
    fun emptyCaptureSpecsOpenSystemGalleryAppWithFallbacks() {
        val specs = buildGalleryLaunchSpecs<String>(
            primaryUri = null,
            mimeType = null,
            recentUris = emptyList(),
        )

        assertEquals(3, specs.size)
        assertEquals(Intent.ACTION_MAIN, specs[0].action)
        assertEquals(Intent.CATEGORY_APP_GALLERY, specs[0].categorySelector)
        assertNull(specs[0].data)
        assertFalse(specs[0].useExternalImagesDir)
        assertTrue(specs[0].newTask)
    }

    @Test
    fun selectRecentPrimaryItemsDeduplicatesStemsAndPrefersJpeg() {
        val items = sequenceOf(
            MediaStoreItem(10L, "PocoPawOpt_1.dng", "image/x-adobe-dng"),
            MediaStoreItem(11L, "PocoPawOpt_1.jpg", "image/jpeg"),
            MediaStoreItem(9L, "PocoPawOpt_0.jpg", "image/jpeg"),
            MediaStoreItem(8L, "PocoPawOpt_old.jpg", "image/jpeg"),
        )
        val selected = selectRecentPrimaryItems(items, limit = 2)
        assertEquals(listOf(11L, 9L), selected.map { it.id })
    }
}
