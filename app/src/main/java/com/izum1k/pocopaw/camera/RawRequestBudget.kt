package com.izum1k.pocopaw.camera

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * RAW requests left before the device must be reopened. marble's CHI layer leaks one RAW target
 * buffer node per RAW request and fails the device at the 66th; only closing and reopening the
 * CameraDevice resets it (docs/research/CAMERA_HAL.md §4.1). Requests without a RAW output are not counted.
 *
 * A single burst of [limit] works on a fresh device, but spread over many captures with preview
 * running in between the device fails a pipeline depth earlier (seen at the 61st), so a device
 * that has already been used keeps [margin] in reserve.
 */
class RawRequestBudget(val limit: Int = DEFAULT_LIMIT, val margin: Int = PIPELINE_DEPTH) {
    private val _remaining = MutableStateFlow(limit)
    val remaining: StateFlow<Int> = _remaining.asStateFlow()

    val consumed: Int get() = if (limit == Int.MAX_VALUE) 0 else (limit - _remaining.value).coerceAtLeast(0)

    fun maxAffordable(): Int {
        val left = _remaining.value
        return if (left == limit) left else (left - margin).coerceAtLeast(0)
    }

    fun canAfford(requests: Int): Boolean = requests <= maxAffordable()

    fun needsProactiveReopen(threshold: Int = PROACTIVE_REOPEN_THRESHOLD): Boolean =
        limit != Int.MAX_VALUE && consumed >= threshold

    fun tryConsume(requests: Int): Boolean {
        if (!canAfford(requests)) return false
        consume(requests)
        return true
    }

    fun consume(requests: Int) {
        check(canAfford(requests)) { "RAW budget exhausted: ${_remaining.value} left, $requests asked" }
        _remaining.value -= requests
    }

    fun recordSubmitted(requests: Int) {
        if (limit == Int.MAX_VALUE || requests <= 0) return
        _remaining.value = (_remaining.value - requests).coerceAtLeast(0)
    }

    fun reset() {
        _remaining.value = limit
    }

    companion object {
        const val DEFAULT_LIMIT = 65
        const val PIPELINE_DEPTH = 8
        const val PROACTIVE_REOPEN_THRESHOLD = 58
    }
}
