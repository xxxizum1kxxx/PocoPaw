package com.izum1k.pocopaw.feature.camera

import android.util.Rational
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.roundToLong

object CameraFormat {
    /** 1/125, 1/4, 0.5", 1", 2.5"; shutter speeds as photographers read them. */
    fun shutter(ns: Long): String {
        val seconds = ns / 1_000_000_000.0
        return if (seconds >= 0.3) {
            val whole = seconds >= 1 && abs(seconds - seconds.roundToLong()) < 0.05
            val s = if (whole) seconds.roundToLong().toString() else String.format(Locale.US, "%.1f", seconds)
            "$s\""
        } else {
            "1/${(1 / seconds).roundToInt()}"
        }
    }

    /** A 35mm-equivalent focal length, as on the lens selector: "24mm". */
    fun equivalentFocal(mm: Float): String = "${mm.roundToInt()}mm"

    fun wb(shift: Int): String = when {
        shift > 0 -> "+$shift"
        shift < 0 -> "−${-shift}"
        else -> "0"
    }

    fun ev(index: Int, step: Rational): String {
        val v = index * step.toFloat()
        if (abs(v) < 0.05f) return "0"
        val sign = if (v > 0) "+" else "−"
        return sign + String.format(Locale.US, "%.1f", abs(v))
    }

    fun zoom(z: Float): String {
        val rounded = (z * 10f).roundToInt() / 10f
        return if (rounded % 1f == 0f) "${rounded.toInt()}×" else String.format(Locale.US, "%.1f×", rounded)
    }

    /** Dioptres as a distance: 12cm, 1.5m, ∞. */
    fun focusDistance(dioptres: Float): String {
        if (dioptres < 0.05f) return "∞"
        val metres = 1f / dioptres
        return if (metres < 1f) "${(metres * 100).roundToInt()}cm" else String.format(Locale.US, "%.1fm", metres)
    }

    fun millimetres(mm: Float): String = String.format(Locale.US, "%.2f mm", mm)
}
