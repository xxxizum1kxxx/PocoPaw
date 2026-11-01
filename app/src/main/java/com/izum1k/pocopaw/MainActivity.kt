package com.izum1k.pocopaw

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation3.runtime.rememberNavBackStack
import com.izum1k.pocopaw.ui.navigation.AppNavHost
import com.izum1k.pocopaw.ui.navigation.Navigator
import com.izum1k.pocopaw.ui.navigation.Route
import com.izum1k.pocopaw.ui.theme.PawTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
        )
        val settingsFlow = container.settings.settings
        setContent {
            val settings by settingsFlow.collectAsStateWithLifecycle()
            PawTheme(settings.colorStyle) {
                val backStack = rememberNavBackStack(Route.Camera)
                val navigator = remember(backStack) { Navigator(backStack) }
                val haptics = LocalHapticFeedback.current
                val feedback = remember(haptics, settings.haptics) { if (settings.haptics) haptics else SilentHaptics }
                CompositionLocalProvider(LocalHapticFeedback provides feedback) {
                    AppNavHost(navigator)
                }
            }
        }
    }
}

private object SilentHaptics : HapticFeedback {
    override fun performHapticFeedback(hapticFeedbackType: HapticFeedbackType) = Unit
}
