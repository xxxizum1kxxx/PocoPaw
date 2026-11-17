package com.izum1k.pocopaw.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RawRequestBudgetTest {

    @Test
    fun freshDeviceAllowsFullLimitAndReservesMarginOnceUsed() {
        val budget = RawRequestBudget()
        assertEquals(65, budget.limit)
        assertEquals(65, budget.remaining.value)
        assertTrue(budget.canAfford(65))
        assertFalse(budget.canAfford(66))

        budget.consume(8)
        assertEquals(57, budget.remaining.value)
        assertTrue(budget.canAfford(49))
        assertFalse(budget.canAfford(50))
    }

    @Test
    fun resetRestoresFullBudgetWithoutMarginPenalty() {
        val budget = RawRequestBudget()
        budget.consume(16)
        budget.reset()
        assertEquals(65, budget.remaining.value)
        assertTrue(budget.canAfford(65))
    }

    @Test(expected = IllegalStateException::class)
    fun consumePastMarginThrows() {
        val budget = RawRequestBudget()
        budget.consume(8)
        budget.consume(50)
    }

    @Test
    fun uncappedBudgetForNonMarbleHasNoMarginLockout() {
        val budget = RawRequestBudget(limit = Int.MAX_VALUE, margin = 0)
        budget.consume(16)
        assertTrue(budget.canAfford(16))
        assertTrue(budget.canAfford(65))
    }
}
