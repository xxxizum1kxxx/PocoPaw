package com.izum1k.pocopaw

import android.app.Application
import android.content.Context
import android.hardware.SensorManager
import android.hardware.camera2.CameraManager
import android.util.Log
import com.izum1k.pocopaw.camera.AwbEstimator
import com.izum1k.pocopaw.camera.AwbEstimatorMode
import com.izum1k.pocopaw.camera.C5AwbEstimator
import com.izum1k.pocopaw.camera.CameraCatalog
import com.izum1k.pocopaw.camera.CameraController
import com.izum1k.pocopaw.camera.FfccAwbEstimator
import com.izum1k.pocopaw.camera.MotionMeter
import com.izum1k.pocopaw.data.SettingsRepository
import com.izum1k.pocopaw.pipeline.LutCatalog
import com.izum1k.pocopaw.pipeline.MediaStoreWriter
import com.izum1k.pocopaw.pipeline.ModelCatalog
import com.izum1k.pocopaw.pipeline.NativeBurst
import com.izum1k.pocopaw.pipeline.ProcessingQueue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class PocoPawApp : Application() {
    val container by lazy { AppContainer(this) }
}

/** Manual dependency injection: one instance of each long-lived object. */
class AppContainer(context: Context) {
    private val appContext = context.applicationContext
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val settings = SettingsRepository(context, scope)
    private val cameraManager = context.getSystemService(CameraManager::class.java)
    val catalog = CameraCatalog(cameraManager)
    val motion = MotionMeter(context.getSystemService(SensorManager::class.java))
    val controller = CameraController(cameraManager, motion).apply {
        // for measuring the analysis stream's cost: `run-as` writes N (0 = off) to files/meter_every
        if (BuildConfig.DEBUG) {
            File(context.filesDir, "meter_every").takeIf { it.isFile }?.readText()?.trim()?.toIntOrNull()?.let { meterEvery = it }
        }
    }
    val output = MediaStoreWriter(context)
    val processing = ProcessingQueue(scope, output)
    private val cacheDir = context.cacheDir.absolutePath
    private val ffccEstimator: AwbEstimator by lazy {
        runCatching { FfccAwbEstimator.fromAssets(appContext.assets, usePrior = false) }
            .getOrElse { e ->
                Log.w(TAG, "FFCC AWB asset load failed, falling back to HAL", e)
                AwbEstimator.HAL
            }
    }
    private val c5Estimator: AwbEstimator by lazy {
        runCatching { C5AwbEstimator.fromAssets(appContext.assets) }
            .getOrElse { e ->
                Log.w(TAG, "C5 AWB asset load failed, falling back to HAL", e)
                AwbEstimator.HAL
            }
    }

    fun pmridModelPath(): String = runCatching { ModelCatalog.pmridModelPath(appContext) }.getOrDefault("")
    fun seaRaftModelPath(): String = runCatching { ModelCatalog.seaRaftModelPath(appContext) }.getOrDefault("")
    fun hdrnetModelPath(): String = runCatching { ModelCatalog.hdrnetModelPath(appContext) }.getOrDefault("")
    fun segModelPath(): String = runCatching { ModelCatalog.segModelPath(appContext) }.getOrDefault("")
    fun lutPathFor(id: String): String = runCatching { LutCatalog.pathFor(appContext, id) }.getOrNull().orEmpty()

    fun checkAssetBytes(relPath: String): Long? = runCatching {
        appContext.assets.openFd(relPath).use { it.length }
    }.recoverCatching {
        appContext.assets.open(relPath).use { input ->
            val buf = ByteArray(8192)
            var total = 0L
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                total += n
            }
            total.takeIf { it > 0L }
        }
    }.getOrNull()

    /** The OpenCL device's properties (or why there is none); creating it compiles the kernels once. */
    val gpuInfo: Deferred<String> by lazy { scope.async { NativeBurst.gpuInit(cacheDir) } }

    init {
        // warm the GPU up before the first accelerated burst
        scope.launch {
            settings.settings.filter { it.gpuAcceleration }.first()
            gpuInfo.await()
        }
        scope.launch {
            settings.settings
                .map { it.awbMode }
                .distinctUntilChanged()
                .collect { mode -> applyAwbMode(mode) }
        }
    }

    fun syncAwbEstimator() {
        scope.launch { applyAwbMode(settings.settings.value.awbMode) }
    }

    private suspend fun applyAwbMode(mode: AwbEstimatorMode) {
        val estimator = when (mode) {
            AwbEstimatorMode.HAL -> AwbEstimator.HAL
            AwbEstimatorMode.FFCC -> withContext(Dispatchers.IO) { ffccEstimator }
            AwbEstimatorMode.C5 -> withContext(Dispatchers.IO) { c5Estimator }
        }
        controller.setAwbEstimator(estimator)
    }

    private companion object {
        const val TAG = "PocoPawApp"
    }
}

val Context.container: AppContainer get() = (applicationContext as PocoPawApp).container
