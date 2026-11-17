package com.izum1k.pocopaw.pipeline

import com.izum1k.pocopaw.camera.ManualExposure
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
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
}
