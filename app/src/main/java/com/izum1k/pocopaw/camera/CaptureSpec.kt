package com.izum1k.pocopaw.camera

import com.izum1k.pocopaw.data.FlashMode

data class ManualExposure(val iso: Int, val exposureNs: Long)

sealed interface ExposurePlan {
    /** AE decides; [evIndex] is in the camera's compensation steps. */
    data class Auto(val evIndex: Int = 0) : ExposurePlan

    data class Manual(val exposure: ManualExposure) : ExposurePlan
}

/**
 * What one shutter press asks for. Only one frame today; the shape is a burst so a multi-frame
 * pipeline can ask for more (up to [RawRequestBudget.DEFAULT_LIMIT] per opened device), and later
 * per-frame exposures for bracketing.
 */
data class CaptureSpec(
    val frameCount: Int = 1,
    val exposure: ExposurePlan = ExposurePlan.Auto(),
    val flash: FlashMode = FlashMode.OFF,
    /** Device rotation at the press, clockwise from natural portrait: 0, 90, 180 or 270. */
    val deviceRotation: Int = 0,
    val zoomRatio: Float = 1f,
) {
    init {
        require(frameCount in 1..RawRequestBudget.DEFAULT_LIMIT) { "frameCount must be 1..${RawRequestBudget.DEFAULT_LIMIT}" }
    }
}

/** Viewfinder controls that shape the repeating preview request. */
data class PreviewControls(
    val exposure: ExposurePlan = ExposurePlan.Auto(),
    val flash: FlashMode = FlashMode.OFF,
    val zoomRatio: Float = 1f,
)

/** What the preview's capture results report, for the viewfinder's readouts. */
data class LiveExposure(
    val iso: Int? = null,
    val exposureNs: Long? = null,
    val focusLocked: Boolean = false,
    val focusing: Boolean = false,
    val aeConverged: Boolean = false,
)
