package com.izum1k.pocopaw.ui.theme

import android.content.Context
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MotionScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.izum1k.pocopaw.data.ColorStyle

/** Seed used when the wallpaper offers none. */
private const val FallbackSeed = 0xFF3F6A2E.toInt()

internal fun appColorScheme(context: Context, style: ColorStyle, wallpaper: SeedPair?): ColorScheme = when {
    style == ColorStyle.EXPRESSIVE && wallpaper != null ->
        ExpressiveColorSchemeFactory.fromSeeds(wallpaper.primary, wallpaper.accent, isDark = true)
    style == ColorStyle.SYSTEM -> dynamicDarkColorScheme(context)
    else -> ExpressiveColorSchemeFactory.fromSeed(FallbackSeed, isDark = true)
}

/**
 * Yoin's design language for a camera: always dark (the picture is the page), M3 Expressive
 * motion, Google Sans Flex, continuous corners, and a wallpaper-seeded scheme whose token
 * changes wash with the effects spring.
 */
@Composable
fun PawTheme(style: ColorStyle = ColorStyle.EXPRESSIVE, content: @Composable () -> Unit) {
    val context = LocalContext.current
    val wallpaper = rememberWallpaperSeeds(enabled = style == ColorStyle.EXPRESSIVE)
    val target = remember(context, style, wallpaper) { appColorScheme(context, style, wallpaper) }
    val motionScheme = MotionScheme.expressive()
    val colorScheme = animateColorScheme(targetColorScheme = target, darkTheme = true, motionScheme = motionScheme)
    MaterialTheme(
        colorScheme = colorScheme,
        motionScheme = motionScheme,
        shapes = YoinShapes,
        typography = YoinTypography,
    ) {
        CompositionLocalProvider(LocalContentColor provides colorScheme.onSurface, content = content)
    }
}
