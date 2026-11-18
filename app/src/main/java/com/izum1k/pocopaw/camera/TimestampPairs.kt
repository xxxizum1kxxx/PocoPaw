package com.izum1k.pocopaw.camera

/**
 * Pairs what an analysis frame's Image gave with what its CaptureResult gave, by sensor
 * timestamp, whichever arrives first. Keeps at most [max] unpaired of each, dropping the oldest.
 */
class TimestampPairs<A : Any, B : Any>(
    private val max: Int,
    private val onDropA: ((A) -> Unit)? = null,
    private val onDropB: ((B) -> Unit)? = null,
) {
    private val images = HashMap<Long, A>()
    private val results = HashMap<Long, B>()

    fun image(ts: Long, a: A): Pair<A, B>? {
        val b = results.remove(ts)
        if (b == null) {
            images.put(ts, a)?.let { prev -> if (prev !== a) onDropA?.invoke(prev) }
            prune()
            return null
        }
        return a to b
    }

    fun result(ts: Long, b: B): Pair<A, B>? {
        val a = images.remove(ts)
        if (a == null) {
            results.put(ts, b)?.let { prev -> if (prev !== b) onDropB?.invoke(prev) }
            prune()
            return null
        }
        return a to b
    }

    fun clear() {
        onDropA?.let { cb -> images.values.forEach(cb) }
        onDropB?.let { cb -> results.values.forEach(cb) }
        images.clear()
        results.clear()
    }

    fun unpairedACount(): Int = images.size
    fun unpairedBCount(): Int = results.size

    private fun prune() {
        while (images.size > max) {
            val k = images.keys.min()
            images.remove(k)?.let { onDropA?.invoke(it) }
        }
        while (results.size > max) {
            val k = results.keys.min()
            results.remove(k)?.let { onDropB?.invoke(it) }
        }
    }
}
