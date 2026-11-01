package com.izum1k.pocopaw.ui.navigation

import androidx.compose.animation.ContentTransform
import androidx.compose.animation.core.FiniteAnimationSpec
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.IntOffset
import androidx.lifecycle.viewmodel.navigation3.rememberViewModelStoreNavEntryDecorator
import androidx.navigation3.runtime.NavKey
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberSaveableStateHolderNavEntryDecorator
import androidx.navigation3.ui.NavDisplay
import com.izum1k.pocopaw.feature.camera.CameraScreen
import com.izum1k.pocopaw.feature.settings.CapabilitiesScreen
import com.izum1k.pocopaw.feature.settings.SettingsScreen
import com.izum1k.pocopaw.ui.theme.YoinMotion

/**
 * One NavDisplay for the whole app. Pushes slide in like the platform's page transition; the
 * system back gesture drives NavDisplay's predictive pop.
 */
@Composable
fun AppNavHost(navigator: Navigator) {
    val motion = PageMotion(
        spatial = YoinMotion.defaultSpatialSpec(),
        effects = YoinMotion.defaultEffectsSpec(),
        fastEffects = YoinMotion.fastEffectsSpec(),
        scale = YoinMotion.defaultSpatialSpec(),
    )
    CompositionLocalProvider(LocalNavigator provides navigator) {
        NavDisplay(
            backStack = navigator.backStack,
            onBack = { navigator.pop() },
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black),
            entryDecorators = listOf(
                rememberSaveableStateHolderNavEntryDecorator<NavKey>(),
                rememberViewModelStoreNavEntryDecorator<NavKey>(),
            ),
            transitionSpec = { motion.push() },
            popTransitionSpec = { motion.pop() },
            predictivePopTransitionSpec = { motion.pop() },
            entryProvider = entryProvider {
                entry<Route.Camera> { CameraScreen() }
                entry<Route.Settings> { SettingsScreen() }
                entry<Route.Capabilities> { CapabilitiesScreen() }
            },
        )
    }
}

private class PageMotion(
    val spatial: FiniteAnimationSpec<IntOffset>,
    val effects: FiniteAnimationSpec<Float>,
    val fastEffects: FiniteAnimationSpec<Float>,
    val scale: FiniteAnimationSpec<Float>,
) {
    fun push(): ContentTransform =
        (slideInHorizontally(spatial) { it / 4 } + fadeIn(effects)) togetherWith
            (slideOutHorizontally(spatial) { -it / 8 } + fadeOut(fastEffects))

    fun pop(): ContentTransform =
        (slideInHorizontally(spatial) { -it / 8 } + fadeIn(effects)) togetherWith
            (scaleOut(scale, targetScale = 0.9f) + fadeOut(fastEffects))
}
