package com.izum1k.pocopaw.camera

/**
 * Pairs what an analysis frame's Image gave with what its CaptureResult gave, by sensor
 * timestamp, whichever arrives first. Keeps at most [max] unpaired of each, dropping the oldest.
 */
class TimestampPairs<A : Any, B : Any>(private val max: Int) {
    private val images = HashMap<Long, A>()
    private val results = HashMap<Long, B>()

    fun image(ts: Long, a: A): Pair<A, B>? {
        val b = results.remove(ts)
        if (b == null) {
            images[ts] = a
            prune()
            return null
        }
        return a to b
    }

    fun result(ts: Long, b: B): Pair<A, B>? {
        val a = images.remove(ts)
        if (a == null) {
            results[ts] = b
            prune()
            return null
        }
        return a to b
    }

    fun clear() {
        images.clear()
        results.clear()
    }

    private fun prune() {
        while (images.size > max) images.remove(images.keys.min())
        while (results.size > max) results.remove(results.keys.min())
    }
}
