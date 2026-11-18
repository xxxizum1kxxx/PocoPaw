package com.izum1k.pocopaw.pipeline

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import java.io.File

/**
 * Accelerator routing policy shared across native burst stages and ONNX Runtime ML sessions.
 * Kept outside [NativeBurst] so JVM unit tests can reference it without loading `libburstpipe.so`.
 */
enum class AccelBackend {
    AUTO,
    CPU,
    GPU_OPENCL,
    NPU_DSP,
    ;

    companion object {
        fun fromOrdinal(v: Int): AccelBackend = entries.getOrElse(v) { AUTO }
    }
}

data class AccelCapabilities(
    val openclAvailable: Boolean = false,
    val openclDevice: String = "",
    val ortAvailable: Boolean = false,
    val ortVersion: String = "",
    val ortProviders: List<String> = emptyList(),
    val nnapiAvailable: Boolean = false,
    val qnnAvailable: Boolean = false,
    val qnnNote: String = "",
    val cdspFastRpcAvailable: Boolean = false,
    val cdspSummary: String = "",
    val mlThreads: Int = 4,
    val bigPrimeCpuIds: List<Int> = emptyList(),
    val hexagonSdkBoundary: String = "",
) {
    companion object {
        fun parse(report: String): AccelCapabilities {
            val map = LinkedHashMap<String, String>()
            for (line in report.lines()) {
                val idx = line.indexOf(": ")
                if (idx > 0) map[line.substring(0, idx).trim()] = line.substring(idx + 2).trim()
            }
            val opencl = map["opencl gpu"].orEmpty()
            val ortVer = map["onnx runtime"].orEmpty()
            val providersRaw = map["ort providers"].orEmpty()
            val providers = if (providersRaw.isEmpty() || providersRaw == "none") emptyList()
            else providersRaw.split(',').map { it.trim() }.filter { it.isNotEmpty() }
            val mlRaw = map["ml threads"].orEmpty()
            val mlThreads = mlRaw.substringBefore(' ').toIntOrNull() ?: 4
            val cpus = Regex("""big\+prime cpus ([0-9,]+)""").find(mlRaw)
                ?.groupValues?.getOrNull(1)
                ?.split(',')
                ?.mapNotNull { it.trim().toIntOrNull() }
                ?: emptyList()
            val cdsp = map["cdsp fastrpc"].orEmpty()
            val qnn = map["qnn ep"].orEmpty()
            return AccelCapabilities(
                openclAvailable = opencl.isNotEmpty() && !opencl.startsWith("unavailable"),
                openclDevice = opencl,
                ortAvailable = ortVer.isNotEmpty() && ortVer != "unavailable",
                ortVersion = ortVer,
                ortProviders = providers,
                nnapiAvailable = map["nnapi ep"] == "available",
                qnnAvailable = qnn == "QNN execution provider available",
                qnnNote = qnn,
                cdspFastRpcAvailable = cdsp.startsWith("available"),
                cdspSummary = cdsp,
                mlThreads = mlThreads,
                bigPrimeCpuIds = cpus,
                hexagonSdkBoundary = map["hexagon sdk boundary"].orEmpty(),
            )
        }
    }
}

/** Shared ONNX Runtime configuration (4 intra-op threads + big+prime core affinity). */
object OrtConfig {
    const val DEFAULT_INTRA_OP_THREADS = 4

    fun detectBigPrimeCpus(
        maxCount: Int = DEFAULT_INTRA_OP_THREADS,
        cpuSysRoot: File = File("/sys/devices/system/cpu"),
    ): List<Int> {
        if (maxCount <= 0 || !cpuSysRoot.isDirectory) return emptyList()
        data class Core(val id: Int, val freqKhz: Long)
        val cores = ArrayList<Core>()
        for (i in 0 until 64) {
            val f = File(cpuSysRoot, "cpu$i/cpufreq/cpuinfo_max_freq")
            if (!f.isFile) continue
            val khz = runCatching { f.readText().trim().toLong() }.getOrNull() ?: continue
            if (khz > 0L) cores += Core(i, khz)
        }
        if (cores.isEmpty()) return emptyList()
        return cores
            .sortedWith(compareByDescending<Core> { it.freqKhz }.thenByDescending { it.id })
            .take(maxCount)
            .map { it.id }
    }

    /** Formats `session.intra_op_thread_affinities` with exact `threads - 1` entries. */
    fun formatIntraOpAffinities(cpus: List<Int>, threads: Int = DEFAULT_INTRA_OP_THREADS): String {
        if (threads <= 1 || cpus.isEmpty()) return ""
        val list = cpus.distinct().sorted().joinToString(",")
        return List(threads - 1) { list }.joinToString(";")
    }

    fun createSessionOptions(
        backend: AccelBackend = AccelBackend.AUTO,
        threads: Int = DEFAULT_INTRA_OP_THREADS,
        bigPrimeCpus: List<Int> = detectBigPrimeCpus(DEFAULT_INTRA_OP_THREADS),
    ): OrtSession.SessionOptions {
        val nThreads = threads.coerceIn(1, DEFAULT_INTRA_OP_THREADS)
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(nThreads)
        opts.setInterOpNumThreads(1)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.addConfigEntry("session.intra_op.allow_spinning", "0")
        val affinities = formatIntraOpAffinities(bigPrimeCpus, nThreads)
        if (affinities.isNotEmpty()) {
            opts.addConfigEntry("session.intra_op_thread_affinities", affinities)
        }
        if (backend == AccelBackend.NPU_DSP) {
            val providers = runCatching { OrtEnvironment.getAvailableProviders() }.getOrDefault(emptySet())
            var appended = false
            if (OrtProvider.QNN in providers) {
                appended = runCatching { opts.addQnn(emptyMap()) }.isSuccess
            }
            if (!appended && OrtProvider.NNAPI in providers) {
                runCatching { opts.addNnapi() }
            }
        }
        return opts
    }
}
