package com.izum1k.pocopaw.feature.camera

import android.content.Intent
import android.provider.MediaStore
import com.izum1k.pocopaw.pipeline.MediaStoreRow
import com.izum1k.pocopaw.pipeline.MediaStoreWriter
import com.izum1k.pocopaw.pipeline.OutputKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GalleryOpenerTest {

    @Test
    fun jpegSpecsPrioritizeReviewWithDeduplicatedClipItemsThenView() {
        val specs = buildGalleryLaunchSpecs(
            primaryUri = "content://media/external/images/media/10",
            mimeType = "image/jpeg",
            recentUris = listOf(
                "content://media/external/images/media/10",
                "content://media/external/images/media/9",
                "content://media/external/images/media/9",
                "content://media/external/images/media/8",
            ),
        )

        assertEquals(6, specs.size)
        assertEquals(
            listOf(
                MediaStore.ACTION_REVIEW to "image/jpeg",
                MediaStore.ACTION_REVIEW to "image/*",
                ACTION_CAMERA_REVIEW_LEGACY to "image/jpeg",
                ACTION_CAMERA_REVIEW_LEGACY to "image/*",
                Intent.ACTION_VIEW to "image/jpeg",
                Intent.ACTION_VIEW to "image/*",
            ),
            specs.map { it.action to it.mimeType },
        )

        val expectedClip = listOf(
            "content://media/external/images/media/9",
            "content://media/external/images/media/8",
        )
        assertEquals(expectedClip, specs[0].clipItems)
        assertEquals(expectedClip, specs[1].clipItems)
        assertEquals(expectedClip, specs[2].clipItems)
        assertEquals(expectedClip, specs[3].clipItems)
        assertTrue(specs[4].clipItems.isEmpty())
        assertTrue(specs[5].clipItems.isEmpty())
        assertTrue(specs.all { it.grantReadUri && it.data == "content://media/external/images/media/10" })
    }

    @Test
    fun dngAndWildcardMimeTypesAvoidDuplicateFallbackEntries() {
        val dngSpecs = buildGalleryLaunchSpecs(
            primaryUri = "content://media/external/images/media/20",
            mimeType = OutputKind.DNG.mimeType,
            recentUris = listOf("content://media/external/images/media/20"),
        )
        assertEquals(
            listOf(
                MediaStore.ACTION_REVIEW to OutputKind.DNG.mimeType,
                MediaStore.ACTION_REVIEW to "image/*",
                ACTION_CAMERA_REVIEW_LEGACY to OutputKind.DNG.mimeType,
                ACTION_CAMERA_REVIEW_LEGACY to "image/*",
                Intent.ACTION_VIEW to OutputKind.DNG.mimeType,
                Intent.ACTION_VIEW to "image/*",
            ),
            dngSpecs.map { it.action to it.mimeType },
        )
        assertTrue(dngSpecs.all { it.clipItems.isEmpty() && it.grantReadUri })

        val wildcardSpecs = buildGalleryLaunchSpecs(
            primaryUri = "content://media/external/images/media/21",
            mimeType = "image/*",
            recentUris = emptyList(),
        )
        assertEquals(
            listOf(
                MediaStore.ACTION_REVIEW to "image/*",
                ACTION_CAMERA_REVIEW_LEGACY to "image/*",
                Intent.ACTION_VIEW to "image/*",
            ),
            wildcardSpecs.map { it.action to it.mimeType },
        )

        val blankMimeSpecs = buildGalleryLaunchSpecs(
            primaryUri = "content://media/external/images/media/22",
            mimeType = "",
            recentUris = emptyList(),
        )
        assertEquals(
            listOf(
                MediaStore.ACTION_REVIEW to "image/*",
                ACTION_CAMERA_REVIEW_LEGACY to "image/*",
                Intent.ACTION_VIEW to "image/*",
            ),
            blankMimeSpecs.map { it.action to it.mimeType },
        )
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
        assertNull(specs[0].mimeType)
        assertFalse(specs[0].useExternalImagesDir)
        assertTrue(specs[0].newTask)

        assertEquals(Intent.ACTION_VIEW, specs[1].action)
        assertEquals(IMAGE_CURSOR_DIR, specs[1].mimeType)
        assertTrue(specs[1].useExternalImagesDir)
        assertTrue(specs[1].newTask)

        assertEquals(Intent.ACTION_VIEW, specs[2].action)
        assertEquals(IMAGE_WILDCARD, specs[2].mimeType)
        assertFalse(specs[2].useExternalImagesDir)
        assertTrue(specs[2].newTask)
    }

    @Test
    fun selectRecentPrimaryRowsDeduplicatesPairsPrefersJpegAndRespectsLimit() {
        val rows = sequenceOf(
            // Pair 1: JPEG first, then DNG
            MediaStoreRow(101L, "PocoPaw_20261118_220000_123.jpg", OutputKind.JPEG.mimeType),
            MediaStoreRow(100L, "PocoPaw_20261118_220000_123.dng", OutputKind.DNG.mimeType),
            // Pair 2: DNG first, then JPEG (must upgrade to JPEG)
            MediaStoreRow(99L, "PocoPaw_20261118_215900_000.dng", OutputKind.DNG.mimeType),
            MediaStoreRow(98L, "PocoPaw_20261118_215900_000.jpg", OutputKind.JPEG.mimeType),
            // Standalone DNG
            MediaStoreRow(97L, "PocoPaw_20261118_215800_500.dng", OutputKind.DNG.mimeType),
            // Extra row beyond limit
            MediaStoreRow(96L, "PocoPaw_20261118_215700_000.jpg", OutputKind.JPEG.mimeType),
        )

        val selected = MediaStoreWriter.selectRecentPrimaryRows(rows, limit = 3)
        assertEquals(listOf(101L, 98L, 97L), selected.map { it.id })
        assertEquals(
            listOf(OutputKind.JPEG.mimeType, OutputKind.JPEG.mimeType, OutputKind.DNG.mimeType),
            selected.map { it.mimeType },
        )

        var consumed = 0
        val countingRows = sequence {
            val all = listOf(
                MediaStoreRow(201L, "PocoPaw_1.dng", OutputKind.DNG.mimeType),
                MediaStoreRow(202L, "PocoPaw_1.jpg", OutputKind.JPEG.mimeType),
                MediaStoreRow(203L, "PocoPaw_2.jpg", OutputKind.JPEG.mimeType),
                MediaStoreRow(204L, "PocoPaw_3.jpg", OutputKind.JPEG.mimeType),
            )
            for (r in all) {
                consumed++
                yield(r)
            }
        }
        val limitOne = MediaStoreWriter.selectRecentPrimaryRows(countingRows, limit = 1)
        assertEquals(listOf(202L), limitOne.map { it.id })
        assertEquals(3, consumed)
        assertTrue(MediaStoreWriter.selectRecentPrimaryRows(rows, limit = 0).isEmpty())
    }
}
