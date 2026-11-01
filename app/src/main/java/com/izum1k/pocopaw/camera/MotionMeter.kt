package com.izum1k.pocopaw.camera

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/**
 * Hand-shake rate for night exposure ("motion metering", as in the probe's night mode): RMS of the
 * gyroscope's angular speed over the last [WINDOW_NS], while [start]ed. Each owner starts and stops
 * it for itself; the sensor runs while any owner holds it.
 */
class MotionMeter(private val sensors: SensorManager?) : SensorEventListener {
    private val gyro = sensors?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val times = LongArray(CAPACITY)
    private val squares = FloatArray(CAPACITY)
    private var head = 0
    private var count = 0
    private var running = false
    private val owners = HashSet<Any>()

    @Synchronized
    fun start(owner: Any = this) {
        owners += owner
        if (running || gyro == null) return
        running = sensors?.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME) == true
    }

    @Synchronized
    fun stop(owner: Any = this) {
        owners -= owner
        if (!running || owners.isNotEmpty()) return
        sensors?.unregisterListener(this)
        running = false
        count = 0
    }

    /** Angular speed RMS in rad/s over the newest [windowNs], or null without a gyroscope or samples. */
    @Synchronized
    fun rms(windowNs: Long = WINDOW_NS): Float? {
        if (count == 0) return null
        val newest = times[(head - 1 + CAPACITY) % CAPACITY]
        var sum = 0.0
        var n = 0
        for (i in 0 until count) {
            val k = (head - 1 - i + CAPACITY) % CAPACITY
            if (newest - times[k] > windowNs) break
            sum += squares[k]
            n++
        }
        return if (n == 0) null else sqrt(sum / n).toFloat()
    }

    /** Sensor time of the newest sample, or null. */
    @Synchronized
    fun newestNs(): Long? = if (count == 0) null else times[(head - 1 + CAPACITY) % CAPACITY]

    /** Angle turned (rad, any axis) since sensor time [sinceNs], as far back as the buffer holds. */
    @Synchronized
    fun angleSince(sinceNs: Long): Float {
        var angle = 0.0
        for (i in 0 until count - 1) {
            val k = (head - 1 - i + CAPACITY) % CAPACITY
            val prev = (k - 1 + CAPACITY) % CAPACITY
            if (times[prev] < sinceNs) break
            angle += sqrt(squares[k].toDouble()) * (times[k] - times[prev]) / 1e9
        }
        return angle.toFloat()
    }

    @Synchronized
    override fun onSensorChanged(event: SensorEvent) {
        val (x, y, z) = Triple(event.values[0], event.values[1], event.values[2])
        times[head] = event.timestamp
        squares[head] = x * x + y * y + z * z
        head = (head + 1) % CAPACITY
        if (count < CAPACITY) count++
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private companion object {
        const val CAPACITY = 256
        const val WINDOW_NS = 1_000_000_000L
    }
}
