package com.izum1k.pocopaw.feature.p1

import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import com.izum1k.pocopaw.R
import com.izum1k.pocopaw.feature.camera.CameraFormat
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt

internal fun evText(ev: Double): String {
    val body = if (abs(ev - ev.roundToInt()) < 1e-6) abs(ev).roundToInt().toString() else String.format(Locale.US, "%.1f", abs(ev))
    return when {
        ev > 1e-6 -> "+$body"
        ev < -1e-6 -> "−$body"
        else -> body
    }
}

/** What one press captures, frame by frame. */
@Composable
internal fun variantDescription(v: P1Variant): String = when {
    v.mode == P1Mode.NIGHT -> stringResource(
        R.string.p1_d_night,
        CameraFormat.shutter(33_000_000L),
        CameraFormat.shutter(v.maxExposureNs),
    )
    v.mode == P1Mode.DARK -> stringResource(
        R.string.p1_d_dark,
        v.totalFrames,
        v.frames,
        v.darkIsos.joinToString("/"),
        CameraFormat.shutter(v.darkExposureNs),
    )
    v.iso != null -> stringResource(R.string.p1_d_iso, v.frames, v.iso, evText(v.ev))
    v.bracket != null -> {
        val stops = List(v.frames) { v.ev + v.bracket[it % v.bracket.size] }
        val counts = stops.groupingBy { it }.eachCount().toList().sortedBy { it.first }
        if (counts.size == 2) {
            stringResource(
                R.string.p1_d_bracket,
                v.frames,
                counts[0].second,
                evText(counts[0].first),
                counts[1].second,
                evText(counts[1].first),
                v.bracket.joinToString(", ") { evText(v.ev + it) },
            )
        } else {
            stringResource(R.string.p1_d_ev, v.frames, v.bracket.joinToString(", ") { evText(v.ev + it) })
        }
    }
    else -> stringResource(R.string.p1_d_ev, v.frames, evText(v.ev))
}

@Composable
internal fun lensLabel(item: P1Item): String =
    stringResource(if (item.cameraId == "2") R.string.p1_lens_uw else R.string.p1_lens_main)
