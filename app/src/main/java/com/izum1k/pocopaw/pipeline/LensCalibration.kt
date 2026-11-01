package com.izum1k.pocopaw.pipeline

/**
 * Per-lens calibrations measured on the Poco F5 (marble) with the research prototype; see
 * docs/research/P1_REPORT.md. Camera ids are the Camera2 ids of this phone.
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
}
