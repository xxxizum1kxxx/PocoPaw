package com.izum1k.pocopaw.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.izum1k.pocopaw.R

val GoogleSansFlex = FontFamily(
    Font(R.font.google_sans_flex_400, FontWeight.Normal),
    Font(R.font.google_sans_flex_500, FontWeight.Medium),
    Font(R.font.google_sans_flex_600, FontWeight.SemiBold),
    Font(R.font.google_sans_flex_700, FontWeight.Bold),
)

private val defaultTypography = Typography()

/**
 * M3 type scale in Google Sans Flex, set solid (Yoin's tracking): 0 across the scale,
 * +0.1sp on the two smallest label styles.
 */
val YoinTypography: Typography = with(defaultTypography) {
    fun TextStyle.flex(tracking: Float = 0f) = copy(fontFamily = GoogleSansFlex, letterSpacing = tracking.sp)
    Typography(
        displayLarge = displayLarge.flex(),
        displayMedium = displayMedium.flex(),
        displaySmall = displaySmall.flex(),
        headlineLarge = headlineLarge.flex(),
        headlineMedium = headlineMedium.flex(),
        headlineSmall = headlineSmall.flex(),
        titleLarge = titleLarge.flex(),
        titleMedium = titleMedium.flex(),
        titleSmall = titleSmall.flex(),
        bodyLarge = bodyLarge.flex(),
        bodyMedium = bodyMedium.flex(),
        bodySmall = bodySmall.flex(),
        labelLarge = labelLarge.flex(),
        labelMedium = labelMedium.flex(0.1f),
        labelSmall = labelSmall.flex(0.1f),
    )
}
