package com.izum1k.pocopaw.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TimestampPairsTest {

    @Test
    fun pairsWhetherImageOrResultArrivesFirst() {
        val pairs = TimestampPairs<String, Int>(max = 4)
        assertNull(pairs.image(100L, "img100"))
        assertEquals("img100" to 1, pairs.result(100L, 1))

        assertNull(pairs.result(200L, 2))
        assertEquals("img200" to 2, pairs.image(200L, "img200"))
    }

    @Test
    fun prunesOldestUnpairedEntriesPastMax() {
        val pairs = TimestampPairs<String, Int>(max = 2)
        assertNull(pairs.image(10L, "a"))
        assertNull(pairs.image(20L, "b"))
        assertNull(pairs.image(30L, "c"))

        assertNull(pairs.result(10L, 1))
        assertEquals("b" to 2, pairs.result(20L, 2))
        assertEquals("c" to 3, pairs.result(30L, 3))
    }

    @Test
    fun clearDropsAllUnpairedEntries() {
        val pairs = TimestampPairs<String, Int>(max = 4)
        pairs.image(50L, "img50")
        pairs.result(60L, 6)
        pairs.clear()

        assertNull(pairs.result(50L, 5))
        assertNull(pairs.image(60L, "img60"))
    }
}
