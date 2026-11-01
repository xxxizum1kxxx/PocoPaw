package com.izum1k.pocopaw.feature.p1

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.izum1k.pocopaw.ui.theme.YoinContainerShapes
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

/** How to hold the phone, then which lens: the same two tags, in the same order, wherever an item is named. */
@Composable
internal fun ItemTags(item: P1Item, modifier: Modifier = Modifier) {
    val c = MaterialTheme.colorScheme
    val (container, content) = when (item.hold) {
        P1Hold.HANDHELD -> c.tertiaryContainer to c.onTertiaryContainer
        P1Hold.BRACED -> c.secondaryContainer to c.onSecondaryContainer
        P1Hold.FIXED -> c.primaryContainer to c.onPrimaryContainer
    }
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Tag(stringResource(item.hold.label), container, content)
        Tag(lensLabel(item), c.surfaceContainerHighest, c.onSurfaceVariant)
    }
}

@Composable
private fun Tag(text: String, container: Color, content: Color) {
    Text(
        text,
        style = MaterialTheme.typography.labelMedium,
        color = content,
        maxLines = 1,
        modifier = Modifier
            .clip(YoinContainerShapes.Card)
            .background(container)
            .padding(horizontal = 8.dp, vertical = 3.dp),
    )
}
