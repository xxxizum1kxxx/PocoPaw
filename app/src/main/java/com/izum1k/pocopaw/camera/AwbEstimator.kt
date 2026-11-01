package com.izum1k.pocopaw.camera

import android.hardware.camera2.params.RggbChannelVector

/**
 * White balance from the analysis stream. Gains it returns are applied to the preview and every
 * still alike (the RAW pipeline reads them back from the result); null leaves the HAL's AWB running.
 */
fun interface AwbEstimator {
    fun estimate(stats: MeterStats): RggbChannelVector?

    companion object {
        val HAL = AwbEstimator { null }
    }
}
