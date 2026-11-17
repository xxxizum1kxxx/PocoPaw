package com.izum1k.pocopaw.feature.camera

import org.junit.Assert.assertEquals
import org.junit.Test

class CoordinateMappingTest {

    @Test
    fun viewAndSensorCoordinatesRoundTripAtOneXAndTwoX() {
        for (orientation in listOf(0, 90, 180, 270)) {
            for (mirrored in listOf(false, true)) {
                for (zoom in listOf(1f, 2f)) {
                    val (cu, cv) = CameraViewModel.viewToSensor(0.5f, 0.5f, orientation, mirrored, zoom)
                    assertEquals(0.5f, cu, 1e-5f)
                    assertEquals(0.5f, cv, 1e-5f)

                    val (u, v) = CameraViewModel.viewToSensor(0.2f, 0.75f, orientation, mirrored, zoom)
                    val (rx, ry) = CameraViewModel.sensorPointToView(u, v, orientation, mirrored, zoom)
                    assertEquals(0.2f, rx, 1e-5f)
                    assertEquals(0.75f, ry, 1e-5f)
                }
            }
        }
    }

    @Test
    fun twoXZoomConfinesViewCornersToCentralQuarterArea() {
        val (u0, v0) = CameraViewModel.viewToSensor(0f, 0f, sensorOrientation = 90, mirrored = false, zoom = 2f)
        val (u1, v1) = CameraViewModel.viewToSensor(1f, 1f, sensorOrientation = 90, mirrored = false, zoom = 2f)
        assertEquals(0.25f, minOf(u0, u1), 1e-5f)
        assertEquals(0.75f, maxOf(u0, u1), 1e-5f)
        assertEquals(0.25f, minOf(v0, v1), 1e-5f)
        assertEquals(0.75f, maxOf(v0, v1), 1e-5f)
    }

    @Test
    fun frontMirroringFlipsHorizontalCoordinateOnly() {
        val unmirrored = CameraViewModel.viewToSensor(0.25f, 0.4f, sensorOrientation = 270, mirrored = false, zoom = 1.5f)
        val mirrored = CameraViewModel.viewToSensor(0.75f, 0.4f, sensorOrientation = 270, mirrored = true, zoom = 1.5f)
        assertEquals(unmirrored.first, mirrored.first, 1e-5f)
        assertEquals(unmirrored.second, mirrored.second, 1e-5f)
    }
}
