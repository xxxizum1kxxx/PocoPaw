package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.camera.ManualExposure
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

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
    fun highlightGuardSmoothlyPullsDownBetween2And18PercentClipped() {
        assertEquals(0.0, BurstExposure.highlightGuardStops(0f), 1e-6)
        assertEquals(0.0, BurstExposure.highlightGuardStops(0.02f), 1e-6)
        assertEquals(-0.5, BurstExposure.highlightGuardStops(0.10f), 1e-5)
        assertEquals(-1.0, BurstExposure.highlightGuardStops(0.18f), 1e-6)
        assertEquals(-1.0, BurstExposure.highlightGuardStops(0.60f), 1e-6)
    }

    @Test
    fun planExposureAppliesHighlightGuardAtZeroEvAndSkipsWhenManualEvSet() {
        val unclipped = BurstExposure.planExposure(
            live = LiveExposure(iso = 200, exposureNs = 16_000_000L, clippedFraction = 0.01f),
            hasOis = true,
            evIndex = 0,
            minIso = 50,
            maxIso = 6400,
            minExp = 10_000L,
        )
        val clipped = BurstExposure.planExposure(
            live = LiveExposure(iso = 200, exposureNs = 16_000_000L, clippedFraction = 0.25f),
            hasOis = true,
            evIndex = 0,
            minIso = 50,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertNotNull(unclipped)
        assertNotNull(clipped)
        // -1 EV total vs -2 EV total at base ISO 50: 32 ms vs 16 ms
        assertEquals(ManualExposure(50, 32_000_000L), unclipped)
        assertEquals(ManualExposure(50, 16_000_000L), clipped)
        assertTrue(clipped!!.exposureNs < unclipped!!.exposureNs)

        // Non-zero manual EV index disables automatic extra highlight pullback so user EV stays predictable
        val manualPlus1 = BurstExposure.planExposure(
            live = LiveExposure(iso = 200, exposureNs = 16_000_000L, clippedFraction = 0.25f),
            hasOis = true,
            evIndex = 3,
            evStep = 1.0 / 3.0,
            minIso = 50,
            maxIso = 6400,
            minExp = 10_000L,
        )
        assertEquals(ManualExposure(50, 64_000_000L), manualPlus1)
    }
}
