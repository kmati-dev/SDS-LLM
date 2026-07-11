"""Tests for OptimizedNGramDrafter."""
from specdecode.simulator.drafter.optimizedNGramDrafter import OptimizedNGramDrafter


def test_optimizedNGramDrafter_instantiation() -> None:
    obj = OptimizedNGramDrafter()
    assert obj is not None
