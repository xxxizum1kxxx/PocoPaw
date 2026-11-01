package com.izum1k.pocopaw

import android.app.Application
import android.content.Context
import android.hardware.SensorManager
import android.hardware.camera2.CameraManager
import com.izum1k.pocopaw.camera.CameraCatalog
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.MotionMeter
import com.izum1k.pocopaw.data.SettingsRepository
import com.izum1k.pocopaw.pipeline.MediaStoreWriter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class PocoPawApp : Application() {
    val container by lazy { AppContainer(this) }
}

/** Manual dependency injection: one instance of each long-lived object. */
class AppContainer(context: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val settings = SettingsRepository(context, scope)
    private val cameraManager = context.getSystemService(CameraManager::class.java)
    val catalog = CameraCatalog(cameraManager)
    val motion = MotionMeter(context.getSystemService(SensorManager::class.java))
    val controller = CameraController(cameraManager, motion)
    val output = MediaStoreWriter(context)
}

val Context.container: AppContainer get() = (applicationContext as PocoPawApp).container
