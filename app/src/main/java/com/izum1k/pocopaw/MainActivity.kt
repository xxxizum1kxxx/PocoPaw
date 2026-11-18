package com.izum1k.pocopaw

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.remember
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
        setContent {
            PawTheme {
                val backStack = rememberNavBackStack(Route.Camera)
                val navigator = remember(backStack) { Navigator(backStack) }
                AppNavHost(navigator)
            }
        }
    }
}
