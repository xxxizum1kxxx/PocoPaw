package com.izum1k.pocopaw.pipeline

/**
 * Per-lens calibrations measured on the Poco F5 (marble) with the research prototype; see
 * docs/research/IMPLEMENTATION.md §2.1 and §3.6. Camera ids are the Camera2 ids of this phone.
 */
object LensCalibration {
    /**
     * Lateral chromatic aberration, from pipeline/calib/lca_cam<id>.json (tools/lca_calib.py). Only the
     * ultrawide has a correction worth applying; the main camera's residual is within the noise.
     */
    private val lca = mapOf(
        "2" to LensLca(kr1 = 0.000471f, kr2 = -0.000348f, kb1 = 0.000096f, kb2 = 0.000372f),
    )

    fun lca(cameraId: String): LensLca? = lca[cameraId]

    /**
     * Sensor noise per ISO, from pipeline/calib/noise_curve_cam<id>.json (tools/noise_calib.py curve: dark and
     * 64-frame tripod captures; ISO 50 from the burst estimates). The ultrawide has no calibration yet and keeps
     * estimating its noise from each burst.
     */
    private val noise = mapOf(
        "0" to NoiseCurve(
            iso = floatArrayOf(50f, 100f, 400f, 800f, 876f),
            a = floatArrayOf(0.06f, 0.11974f, 0.49157f, 0.98734f, 1.24656f),
            b = floatArrayOf(0.3f, 0.4885f, 2.0218f, 5.2842f, 5.3586f),
        ),
    )

    fun noise(cameraId: String): NoiseCurve? = noise[cameraId]
}
