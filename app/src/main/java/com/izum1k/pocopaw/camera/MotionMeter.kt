package com.izum1k.pocopaw.camera

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/**
 * Hand-shake rate for night exposure ("motion metering", as in the probe's night mode): RMS of the
 * gyroscope's angular speed over the last [WINDOW_NS], while [start]ed.
 */
class MotionMeter(private val sensors: SensorManager?) : SensorEventListener {
    private val gyro = sensors?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val times = LongArray(CAPACITY)
    private val squares = FloatArray(CAPACITY)
    private var head = 0
    private var count = 0
    private var running = false

    @Synchronized
    fun start() {
        if (running || gyro == null) return
        running = sensors?.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME) == true
    }

    @Synchronized
    fun stop() {
        if (!running) return
        sensors?.unregisterListener(this)
        running = false
        count = 0
    }

    /** Angular speed RMS in rad/s, or null without a gyroscope or samples. */
    @Synchronized
    fun rms(): Float? {
        if (count == 0) return null
        val newest = times[(head - 1 + CAPACITY) % CAPACITY]
        var sum = 0.0
        var n = 0
        for (i in 0 until count) {
            val k = (head - 1 - i + CAPACITY) % CAPACITY
            if (newest - times[k] > WINDOW_NS) break
            sum += squares[k]
            n++
        }
        return if (n == 0) null else sqrt(sum / n).toFloat()
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
