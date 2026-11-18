package com.izum1k.pocopaw

import android.app.Application
import android.content.Context
import android.hardware.SensorManager
import android.hardware.camera2.CameraManager
import com.izum1k.pocopaw.camera.CameraCatalog
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.MotionMeter
import com.izum1k.pocopaw.data.DebugCapture
import com.izum1k.pocopaw.data.SettingsRepository
import com.izum1k.pocopaw.pipeline.MediaStoreWriter
import com.izum1k.pocopaw.pipeline.NativeBurst
import com.izum1k.pocopaw.pipeline.ProcessingQueue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import java.io.File

class PocoPawApp : Application() {
    val container by lazy { AppContainer(this) }
}

/** Manual dependency injection: one instance of each long-lived object. */
class AppContainer(context: Context) {
    private val appContext = context.applicationContext
    val packageName: String = appContext.packageName
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val settings = SettingsRepository(appContext, scope)
    private val cameraManager = appContext.getSystemService(CameraManager::class.java)
    val catalog = CameraCatalog(cameraManager)
    val motion = MotionMeter(appContext.getSystemService(SensorManager::class.java))
    val controller = CameraController(cameraManager, motion).apply {
        if (BuildConfig.DEBUG) {
            File(appContext.filesDir, "meter_every").takeIf { it.isFile }?.readText()?.trim()?.toIntOrNull()?.let { meterEvery = it }
        }
    }
    val output = MediaStoreWriter(appContext)
    val processing = ProcessingQueue(scope, output)
    val debugCapturesDir: File by lazy { DebugCapture.rootDir(appContext) }
    private val cacheDir = appContext.cacheDir.absolutePath

    /** Pre-warms OpenCL kernel compilation and binary cache off the main thread. */
    val gpuInfo: Deferred<String> by lazy { scope.async { NativeBurst.gpuInit(cacheDir) } }

    init {
        scope.launch { gpuInfo.await() }
    }
}

val Context.container: AppContainer get() = (applicationContext as PocoPawApp).container
