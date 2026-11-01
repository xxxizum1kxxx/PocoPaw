package com.izum1k.pocopaw.feature.p1

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import com.izum1k.pocopaw.camera.CameraInfo
import com.izum1k.pocopaw.camera.LiveExposure
import com.izum1k.pocopaw.pipeline.RawFrame
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Where the captures live: `<root>/<item>/<item>[_<variant>]_<yyyyMMdd_HHmmss>/`, one burst per folder. */
class P1Store(context: Context) {
    val root: File = File(context.getExternalFilesDir(null), "p1")

    fun itemDir(item: P1Item) = File(root, item.name)

    /** Finished bursts of [item] (the folders with a `meta.json`), oldest first. */
    fun sets(item: P1Item): List<File> =
        itemDir(item).listFiles()?.filter { File(it, META).isFile }?.sortedBy { it.name }.orEmpty()

    fun count(item: P1Item, variant: P1Variant): Int =
        if (item.variants.size == 1) sets(item).size else sets(item).count { it.name.startsWith("${item.name}_${variant.key}_") }

    fun newSet(item: P1Item, variant: P1Variant, time: Date = Date()): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(time)
        val base = if (item.variants.size == 1) "${item.name}_$stamp" else "${item.name}_${variant.key}_$stamp"
        var dir = File(itemDir(item), base)
        var k = 2
        while (dir.exists()) dir = File(itemDir(item), "${base}_${k++}")
        return dir.apply { mkdirs() }
    }

    fun clear() = root.deleteRecursively()

    companion object {
        const val META = "meta.json"
    }
}

/**
 * Writes one burst exactly as probe's `CaptureActivity` does: `frameNN.raw16` (16-bit little
 * endian, rows tightly packed) per frame in capture order, then [finish] writes `meta.json`.
 */
class ProbeWriter(val dir: File) {
    private data class Shot(val exposureNs: Long?, val iso: Int?, val timestampNs: Long)

    private val shots = mutableListOf<Shot>()
    private var first: CaptureResult? = null
    private var width = 0
    private var height = 0

    val written: Int get() = shots.size
    val firstIso: Int? get() = shots.firstOrNull()?.iso
    val firstExposureNs: Long? get() = shots.firstOrNull()?.exposureNs

    fun write(frame: RawFrame) {
        FileOutputStream(File(dir, String.format(Locale.US, "frame%02d.raw16", shots.size))).use { out ->
            out.channel.write(frame.data.duplicate().apply { rewind() })
        }
        if (first == null) {
            first = frame.result
            width = frame.width
            height = frame.height
        }
        shots += Shot(frame.exposureNs, frame.iso, frame.timestampNs)
    }

    fun finish(item: P1Item, variant: P1Variant, cam: CameraInfo, meter: LiveExposure, gyroRms: Double) {
        val r0 = first ?: return
        val cc = cam.characteristics
        val j = StringBuilder("{\n")
        fun kv(k: String, v: Any?) {
            j.append("  \"").append(k).append("\": ").append(v ?: "null").append(",\n")
        }
        kv("camera", "\"${cam.id}\"")
        kv("width", width)
        kv("height", height)
        kv("rowStride", width * 2)
        kv("cfa", cc.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT))
        val black = r0.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)?.map { Math.round(it) }
            ?: cc.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)?.let { p -> List(4) { i -> p.getOffsetForIndex(i % 2, i / 2) } }
            ?: List(4) { cam.blackLevel }
        kv("blackLevel", black.joinToString(", ", "[", "]"))
        kv("whiteLevel", r0.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL) ?: cc.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL))
        val g = r0.get(CaptureResult.COLOR_CORRECTION_GAINS)
        kv("wbGains", g?.let { String.format(Locale.US, "[%f, %f, %f, %f]", it.red, it.greenEven, it.greenOdd, it.blue) })
        kv("ccm", matrix(r0.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)))
        val neu = r0.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)
        kv("neutral", neu?.let { String.format(Locale.US, "[%f, %f, %f]", it[0].toDouble(), it[1].toDouble(), it[2].toDouble()) })
        kv("forwardMatrix1", matrix(cc.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX1)))
        kv("forwardMatrix2", matrix(cc.get(CameraCharacteristics.SENSOR_FORWARD_MATRIX2)))
        kv("colorTransform1", matrix(cc.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM1)))
        kv("colorTransform2", matrix(cc.get(CameraCharacteristics.SENSOR_COLOR_TRANSFORM2)))
        kv("illuminant1", cc.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1))
        kv("illuminant2", cc.get(CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2) ?: -1)
        r0.get(CaptureResult.STATISTICS_LENS_SHADING_CORRECTION_MAP)?.let { lsm ->
            kv("lscRows", lsm.rowCount)
            kv("lscCols", lsm.columnCount)
            val gains = FloatArray(lsm.gainFactorCount).also { lsm.copyGainFactors(it, 0) }
            kv("lscGains", gains.joinToString(",", "[", "]") { String.format(Locale.US, "%.4f", it) })
        }
        kv("meterExposureNs", meter.exposureNs)
        kv("meterIso", meter.iso)
        kv("ev", variant.ev)
        kv("mode", "\"${variant.mode.id}\"")
        kv("gyroRms", String.format(Locale.US, "%.5f", gyroRms))
        variant.bracket?.let { kv("bracket", it.joinToString(", ", "[", "]")) }
        kv("item", "\"${item.name}\"")
        kv("variant", "\"${variant.key}\"")
        j.append("  \"frames\": [")
        shots.forEachIndexed { i, s ->
            if (i > 0) j.append(",\n    ")
            j.append("{\"exposureNs\": ${s.exposureNs}, \"iso\": ${s.iso}, \"timestampNs\": ${s.timestampNs}}")
        }
        j.append("]\n}\n")
        File(dir, P1Store.META).writeText(j.toString())
    }

    private fun matrix(t: ColorSpaceTransform?): String? = t?.let {
        (0 until 9).joinToString(", ", "[", "]") { k -> String.format(Locale.US, "%f", t.getElement(k % 3, k / 3).toDouble()) }
    }
}
