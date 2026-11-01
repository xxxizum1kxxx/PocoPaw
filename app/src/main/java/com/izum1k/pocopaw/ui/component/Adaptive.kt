package com.izum1k.pocopaw.ui.component

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.isSpecified

/** Yoin's three layout tiers (Compact < 600 ≤ Medium < 840 ≤ Expanded). */
enum class WidthClass { Compact, Medium, Expanded }

/** Width of the pane a page is laid out in; unspecified means the whole window. */
val LocalLayoutWidth = staticCompositionLocalOf { Dp.Unspecified }

@Composable
fun widthClass(): WidthClass {
    val pane = LocalLayoutWidth.current
    val w = if (pane.isSpecified) pane.value.toInt() else LocalConfiguration.current.screenWidthDp
    return when {
        w < 600 -> WidthClass.Compact
        w < 840 -> WidthClass.Medium
        else -> WidthClass.Expanded
    }
}

/** Centers reading-width content on wide windows; nothing stretches into a 1280dp column. */
fun Modifier.readingWidth(max: Dp = 840.dp): Modifier =
    this.fillMaxWidth().wrapContentWidth(Alignment.CenterHorizontally).widthIn(max = max)

/** Space the shell's floating bar occupies at the bottom (0 on pushed pages). */
val LocalBottomBarInset = staticCompositionLocalOf { 0.dp }
