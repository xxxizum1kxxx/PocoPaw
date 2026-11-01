package com.izum1k.pocopaw.feature.p1

import androidx.annotation.StringRes
import com.izum1k.pocopaw.R

/** The probe's capture modes, as written to `meta.json`. */
enum class P1Mode(val id: String) { HDRPLUS("hdrplus"), NIGHT("night"), DARK("dark") }

/**
 * What one shutter press captures, with the probe's semantics: a manual RAW burst at the preview's
 * metered exposure × 2^([ev] + [bracket] cycled), at the lowest ISO whose shutter stays within
 * [maxExposureNs]; [iso] fixes the ISO for every frame instead. [P1Mode.NIGHT] caps the shutter
 * by gyro motion metering and picks the frame count for about 3 s; [P1Mode.DARK] shoots [frames]
 * at each of [darkIsos] with a fixed [darkExposureNs].
 */
data class P1Variant(
    /** Folder name part; unique within an item. */
    val key: String,
    val mode: P1Mode = P1Mode.HDRPLUS,
    /** Frames per press; 0 for night, where motion metering decides. */
    val frames: Int,
    val ev: Double = -1.0,
    val bracket: List<Double>? = null,
    val iso: Int? = null,
    val maxExposureNs: Long = if (mode == P1Mode.NIGHT) 500_000_000L else 66_000_000L,
    val darkIsos: List<Int> = emptyList(),
    val darkExposureNs: Long = 10_000_000L,
) {
    /** Frames the press asks for, where known before metering. */
    val totalFrames: Int get() = if (mode == P1Mode.DARK) minOf(64, frames * darkIsos.size) else frames
}

enum class P1Group(@StringRes val title: Int, @StringRes val note: Int) {
    MOTION(R.string.p1_group_motion, R.string.p1_note_motion),
    BRACKET(R.string.p1_group_bracket, R.string.p1_note_bracket),
    NIGHT(R.string.p1_group_night, R.string.p1_note_night),
    AWB(R.string.p1_group_awb, R.string.p1_note_awb),
    TRIPOD(R.string.p1_group_tripod, R.string.p1_note_tripod),
    DARK(R.string.p1_group_dark, R.string.p1_note_dark),
    LCA(R.string.p1_group_lca, R.string.p1_note_lca),
}

/** One row of CAPTURE_P1.md. [variants] first is what the document asks for; the rest are optional extras. */
data class P1Item(
    val name: String,
    val group: P1Group,
    val cameraId: String,
    @StringRes val scene: Int?,
    val variants: List<P1Variant>,
) {
    val required: P1Variant get() = variants.first()
}

object P1Items {
    private fun standard(n: Int) = listOf(
        P1Variant("ev-1", frames = n, ev = -1.0),
        P1Variant("ev0", frames = n, ev = 0.0),
        P1Variant("ev-2", frames = n, ev = -2.0),
    )

    private fun bracket() = listOf(
        P1Variant("bracket", frames = 12, ev = -2.0, bracket = listOf(0.0, 0.0, 2.0)),
        P1Variant("ev0", frames = 12, ev = 0.0),
        P1Variant("ev-2", frames = 12, ev = -2.0),
    )

    private val night = listOf(P1Variant("night", mode = P1Mode.NIGHT, frames = 0, ev = 0.0))

    val all: List<P1Item> = listOf(
        P1Item("mo_main_wave", P1Group.MOTION, "0", R.string.p1_scene_wave, standard(8)),
        P1Item("mo_main_walk", P1Group.MOTION, "0", R.string.p1_scene_walk, standard(8)),
        P1Item("mo_uw_street", P1Group.MOTION, "2", R.string.p1_scene_street, standard(8)),
        P1Item("br_main_backlit", P1Group.BRACKET, "0", null, bracket()),
        P1Item("br_uw_backlit", P1Group.BRACKET, "2", null, bracket()),
        P1Item("nt_main_hand", P1Group.NIGHT, "0", R.string.p1_scene_hand, night),
        P1Item("nt_main_brace", P1Group.NIGHT, "0", R.string.p1_scene_brace, night),
        P1Item("nt_uw_hand", P1Group.NIGHT, "2", R.string.p1_scene_hand, night),
        P1Item("awb_day", P1Group.AWB, "0", R.string.p1_scene_day, standard(4)),
        P1Item("awb_shade", P1Group.AWB, "0", R.string.p1_scene_shade, standard(4)),
        P1Item("awb_warm", P1Group.AWB, "0", R.string.p1_scene_warm, standard(4)),
        P1Item("awb_cool", P1Group.AWB, "0", R.string.p1_scene_cool, standard(4)),
        P1Item("awb_mixed", P1Group.AWB, "0", R.string.p1_scene_mixed, standard(4)),
        P1Item("awb_night", P1Group.AWB, "0", R.string.p1_scene_street_light, standard(4)),
        P1Item(
            "tp_main_iso100", P1Group.TRIPOD, "0", null,
            listOf(P1Variant("iso100", frames = 64, iso = 100, maxExposureNs = 250_000_000L)),
        ),
        P1Item("tp_main_iso800", P1Group.TRIPOD, "0", null, listOf(P1Variant("iso800", frames = 64, iso = 800, ev = -2.0))),
        P1Item("tp_main_iso3200", P1Group.TRIPOD, "0", null, listOf(P1Variant("iso3200", frames = 64, iso = 3200, ev = -4.0))),
        P1Item(
            "dk_main", P1Group.DARK, "0", null,
            listOf(P1Variant("dark", mode = P1Mode.DARK, frames = 8, darkIsos = listOf(100, 400, 800, 1600, 3200, 6400))),
        ),
        P1Item("lca_main", P1Group.LCA, "0", null, standard(8)),
        P1Item("lca_uw", P1Group.LCA, "2", null, standard(8)),
    )

    fun byName(name: String): P1Item? = all.firstOrNull { it.name == name }
}
