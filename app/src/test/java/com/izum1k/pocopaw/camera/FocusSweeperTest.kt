package com.izum1k.pocopaw.camera

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FocusSweeperTest {

    @Test
    fun centreZonesAreSortedHighestWeightFirst() {
        val zones = FocusSweeper.centreZones()
        assertEquals(9, zones.size)
        assertEquals(listOf(4f, 2f, 2f, 2f, 2f, 1f, 1f, 1f, 1f), zones.map { it.weight })
    }

    @Test
    fun coarseAndFineStepsCoverRangeToMinFocus() {
        assertEquals(
            listOf(0f, 0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.5f, 4f, 5f, 6.5f, 8f, 10f),
            FocusSweeper.coarseSteps(10f),
        )
        assertEquals(
            listOf(0f, 0.5f, 1f, 1.5f, 2f, 2.5f, 3f, 3.2f),
            FocusSweeper.coarseSteps(3.2f),
        )
        val fine = FocusSweeper.fineSteps(lo = 2f, hi = 5f, step = 0.5f, minFocusDioptres = 10f)
        assertEquals(7, fine.size)
        assertEquals(2f, fine.first(), 1e-5f)
        assertEquals(5f, fine.last(), 1e-5f)
    }

    @Test
    fun pickFavorsCentreZoneWhenTwoPeaksAreComparable() {
        val zones = listOf(FocusZone(RectF(), weight = 4f), FocusZone(RectF(), weight = 2f))
        val samples = listOf(
            FocusSample(0, 0f, 0f, true, floatArrayOf(10f, 10f)),
            FocusSample(1, 1f, 1f, true, floatArrayOf(12f, 30f)),
            FocusSample(2, 2f, 2f, true, floatArrayOf(10f, 10f)),
            FocusSample(3, 3f, 3f, true, floatArrayOf(30f, 18f)),
            FocusSample(4, 4f, 4f, true, floatArrayOf(10f, 10f)),
        )
        assertEquals(3, FocusSweeper.pick(samples, zones))
        assertTrue(FocusSweeper.contrast(samples, zones) > FocusSweeper.FLAT_CONTRAST)
    }

    @Test
    fun fitPeakAndInterpolateRecoverQuadraticVertex() {
        val zones = listOf(FocusZone(RectF(), weight = 1f))
        val target = 3.25f
        val samples = listOf(2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f).mapIndexed { i, d ->
            val score = 100f - 10f * (d - target) * (d - target)
            FocusSample(i, d, d, true, floatArrayOf(score))
        }
        val fitted = FocusSweeper.fitPeak(samples, zones)
        assertNotNull(fitted)
        assertEquals(target, fitted!!, 1e-3f)
        assertEquals(target, FocusSweeper.interpolate(samples, zones, 2), 1e-3f)

        val upward = samples.mapIndexed { i, s -> FocusSample(i, s.target, s.dioptres, true, floatArrayOf(200f - s.scores[0])) }
        assertNull(FocusSweeper.fitPeak(upward, zones))
        assertNull(FocusSweeper.fitPeak(samples.take(3), zones))
    }
}
