package com.izum1k.pocopaw.ui.navigation

import androidx.compose.runtime.staticCompositionLocalOf
import androidx.navigation3.runtime.NavBackStack
import androidx.navigation3.runtime.NavKey
import kotlinx.serialization.Serializable

/** Every destination. [Camera] is the root; settings pages are pushed over it. */
@Serializable
sealed interface Route : NavKey {
    @Serializable data object Camera : Route

    @Serializable data object Settings : Route

    /** RAW specifications of every usable camera. */
    @Serializable data object Capabilities : Route

    /** The P1 capture list (docs/research/VALIDATION.md appendix A), for collecting the offline pipeline's test material. */
    @Serializable data object P1Captures : Route

    /** One P1 capture list item, by its name. */
    @Serializable data class P1Capture(val item: String) : Route
}

class Navigator(internal val backStack: NavBackStack<NavKey>) {
    fun push(route: Route) {
        backStack.add(route)
    }

    fun pop(): Boolean {
        if (backStack.size <= 1) return false
        backStack.removeAt(backStack.lastIndex)
        return true
    }
}

val LocalNavigator = staticCompositionLocalOf<Navigator> { error("No Navigator provided") }
