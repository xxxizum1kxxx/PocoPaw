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

    @Test
    fun errorRecoveryClearDropsUnpairedTimestampPairsWithCleanup() {
        val dropped = mutableListOf<String>()
        val pairs = TimestampPairs<String, Int>(max = 2, onDropA = { dropped += it })
        pairs.image(1L, "buf1")
        pairs.image(2L, "buf2")
        pairs.image(3L, "buf3")
        assertEquals(listOf("buf1"), dropped)

        pairs.clear()
        assertEquals(listOf("buf1", "buf2", "buf3"), dropped)
        assertEquals(0, pairs.unpairedACount())
        assertEquals(0, pairs.unpairedBCount())
    }

    @Test
    fun duplicateTimestampOverwriteInvokesDropCallbacks() {
        val droppedA = mutableListOf<String>()
        val droppedB = mutableListOf<String>()
        val pairs = TimestampPairs<String, String>(
            max = 4,
            onDropA = { droppedA += it },
            onDropB = { droppedB += it },
        )

        assertNull(pairs.image(10L, "buf1"))
        assertNull(pairs.image(10L, "buf2"))
        assertEquals(listOf("buf1"), droppedA)
        assertEquals("buf2" to "res10", pairs.result(10L, "res10"))

        assertNull(pairs.result(20L, "resA"))
        assertNull(pairs.result(20L, "resB"))
        assertEquals(listOf("resA"), droppedB)
        assertEquals("buf20" to "resB", pairs.image(20L, "buf20"))
    }
}
