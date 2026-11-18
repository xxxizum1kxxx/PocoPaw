package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.pow
import kotlin.math.roundToLong

class BurstExposureTest {

    @Test
    fun underexposureDropsIsoToBaseBeforeRaisingExposure() {
        val m = BurstExposure.meterExposure(
            iso0 = 400,
            exp0 = 10_000_000L,
            stops = -1.0,
            limit = 66_000_000L,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertNotNull(m)
        assertEquals(100, m!!.iso)
        assertEquals(20_000_000L, m.exposureNs)
    }

    @Test
    fun shutterCapRollsOverRemainingExposureToIso() {
        val m = BurstExposure.meterExposure(
            iso0 = 800,
            exp0 = 30_000_000L,
            stops = 0.0,
            limit = 33_000_000L,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertNotNull(m)
        assertEquals(33_000_000L, m!!.exposureNs)
        assertEquals(728, m.iso)
    }

    @Test
    fun clampsToMaxIsoAndMinExposureBounds() {
        val over = BurstExposure.meterExposure(
            iso0 = 6400,
            exp0 = 66_000_000L,
            stops = 1.0,
            limit = 33_000_000L,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertEquals(ManualExposure(6400, 33_000_000L), over)

        val under = BurstExposure.meterExposure(
            iso0 = 50,
            exp0 = 20_000L,
            stops = -2.0,
            limit = 33_000_000L,
            minIso = 50,
            maxIso = 3200,
            minExp = 10_000L,
        )
        assertEquals(ManualExposure(50, 10_000L), under)
    }

    @Test
    fun invalidLiveInputsReturnNull() {
        assertNull(BurstExposure.meterExposure(iso0 = 0, exp0 = 10_000_000L, stops = 0.0, limit = 33_000_000L))
        assertNull(BurstExposure.meterExposure(iso0 = 100, exp0 = 0L, stops = 0.0, limit = 33_000_000L))
        assertNull(BurstExposure.meterExposure(iso0 = 100, exp0 = 10_000_000L, stops = 0.0, limit = 0L))
    }

    @Test
    fun highlightGuardIsSmoothBoundedAndSafeOnInvalidValues() {
        assertEquals(0.0, BurstExposure.highlightGuardStops(0f), 1e-9)
        assertEquals(0.0, BurstExposure.highlightGuardStops(-0.1f), 1e-9)
        assertEquals(0.0, BurstExposure.highlightGuardStops(Float.NaN), 1e-9)
        assertEquals(0.0, BurstExposure.highlightGuardStops(Float.POSITIVE_INFINITY), 1e-9)
        assertEquals(0.0, BurstExposure.highlightGuardStops(BurstExposure.HIGHLIGHT_CLIP_SAFE), 1e-9)

        val mid = 0.5f * (BurstExposure.HIGHLIGHT_CLIP_SAFE + BurstExposure.HIGHLIGHT_CLIP_FULL)
        assertEquals(-0.5 * BurstExposure.HIGHLIGHT_GUARD_MAX_STOPS, BurstExposure.highlightGuardStops(mid), 1e-6)

        val a = BurstExposure.highlightGuardStops(0.05f)
        val b = BurstExposure.highlightGuardStops(0.10f)
        val c = BurstExposure.highlightGuardStops(0.15f)
        assertTrue(a < 0.0 && b < a && c < b && c > -BurstExposure.HIGHLIGHT_GUARD_MAX_STOPS)

        assertEquals(-BurstExposure.HIGHLIGHT_GUARD_MAX_STOPS, BurstExposure.highlightGuardStops(BurstExposure.HIGHLIGHT_CLIP_FULL), 1e-9)
        assertEquals(-BurstExposure.HIGHLIGHT_GUARD_MAX_STOPS, BurstExposure.highlightGuardStops(1.0f), 1e-9)
    }

    @Test
    fun planExposureProtectsHighlightsOnlyWhenEvIndexIsZero() {
        val noClip = BurstExposure.planExposure(
            live = LiveExposure(iso = 400, exposureNs = 10_000_000L, clippedFraction = 0.01f),
            hasOis = true,
            evIndex = 0,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertEquals(ManualExposure(100, 20_000_000L), noClip)

        val midClip = BurstExposure.planExposure(
            live = LiveExposure(iso = 400, exposureNs = 10_000_000L, clippedFraction = 0.10f),
            hasOis = true,
            evIndex = 0,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        val expectedMidExp = (400.0 * 10_000_000L * 2.0.pow(-1.5) / 100.0).roundToLong()
        assertNotNull(midClip)
        assertEquals(100, midClip!!.iso)
        assertTrue(kotlin.math.abs(midClip.exposureNs - expectedMidExp) <= 10L)

        val fullClip = BurstExposure.planExposure(
            live = LiveExposure(iso = 400, exposureNs = 10_000_000L, clippedFraction = 0.50f),
            hasOis = true,
            evIndex = 0,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertEquals(ManualExposure(100, 10_000_000L), fullClip)

        // Manual EV steps remain exact and unaffected by preview clipping.
        val manualPlus = BurstExposure.planExposure(
            live = LiveExposure(iso = 400, exposureNs = 10_000_000L, clippedFraction = 0.50f),
            hasOis = true,
            evIndex = 3,
            evStep = 1.0 / 3.0,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertEquals(ManualExposure(100, 40_000_000L), manualPlus)

        val manualMinus = BurstExposure.planExposure(
            live = LiveExposure(iso = 400, exposureNs = 10_000_000L, clippedFraction = 0.50f),
            hasOis = false,
            evIndex = -1,
            evStep = 0.5,
            minIso = 100,
            maxIso = 6400,
            minExp = 10_000L,
        )
        val expectedMinusExp = (400.0 * 10_000_000L * 2.0.pow(-1.5) / 100.0).roundToLong()
        assertEquals(ManualExposure(100, expectedMinusExp), manualMinus)

        assertNull(BurstExposure.planExposure(LiveExposure(iso = null, exposureNs = 10_000_000L, clippedFraction = 0.5f), hasOis = true))
    }

    @Test
    fun twoXZoomBoostScalesFrameCountBetween12And16() {
        assertEquals(8, BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 1.0f))
        assertEquals(4, BurstExposure.framesForZoom(baseFrames = 4, cropZoom = 1.0f))

        val midBoost = BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 1.5f)
        assertTrue(midBoost in 12..16)
        assertEquals(14, midBoost)

        val lowBoost = BurstExposure.framesForZoom(baseFrames = 4, cropZoom = 1.05f)
        assertTrue(lowBoost in 12..16)

        assertEquals(16, BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 2.0f))
        assertEquals(16, BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 2.5f))
        assertEquals(1, BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 2.0f, flash = true))
    }

    @Test
    fun lowBudgetClampsZoomBoostSafelyWithoutException() {
        val budget = com.izum1k.pocopaw.camera.RawRequestBudget()
        budget.consume(50) // 15 remaining, 8 margin -> maxAffordable = 7
        val desired = BurstExposure.framesForZoom(baseFrames = 8, cropZoom = 2.0f)
        assertEquals(16, desired)
        val clamped = BurstExposure.framesWithinBudget(desired, budget)
        assertEquals(7, clamped)
        assertTrue(budget.canAfford(clamped))
    }

    @Test
    fun framesWithinBudgetReturnsZeroWhenBudgetExhausted() {
        val budget = com.izum1k.pocopaw.camera.RawRequestBudget()
        budget.recordSubmitted(65)
        assertEquals(0, budget.maxAffordable())
        val clamped = BurstExposure.framesWithinBudget(16, budget)
        assertEquals(0, clamped)
        assertTrue(clamped <= budget.maxAffordable())
        assertTrue(budget.canAfford(clamped))
    }
}
