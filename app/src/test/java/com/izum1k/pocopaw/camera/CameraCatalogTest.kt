package com.izum1k.pocopaw.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraCatalogTest {

    @Test
    fun cameraCatalogDenylistBlocksAuxiliaryAndMacroIdsAndAdmitsPhysicalIds() {
        val candidates = listOf("0", "1", "2", "3", "4", "5", "6")
        val admitted = candidates.filterNot { it in CameraCatalog.DENYLIST }
        assertEquals(listOf("0", "1", "2"), admitted)
        assertTrue("3" in CameraCatalog.DENYLIST)
        assertTrue("4" in CameraCatalog.DENYLIST)
        assertTrue("5" in CameraCatalog.DENYLIST)
        assertTrue("6" in CameraCatalog.DENYLIST)
        assertFalse("0" in CameraCatalog.DENYLIST)
        assertFalse("1" in CameraCatalog.DENYLIST)
        assertFalse("2" in CameraCatalog.DENYLIST)
    }
}
