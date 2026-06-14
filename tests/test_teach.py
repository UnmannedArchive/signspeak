"""Tests for the offline (no-API) English->gloss extraction."""
from asl.teach import heuristic_glosses


def test_heuristic_drops_function_words_keeps_order():
    glosses = [g for g, _ in heuristic_glosses("I want to drink the water")]
    assert glosses == ["i", "want", "drink", "water"]  # "to", "the" dropped


def test_heuristic_handles_punctuation_and_case():
    glosses = [g for g, _ in heuristic_glosses("Hello, are you DEAF?")]
    assert glosses == ["hello", "you", "deaf"]  # "are" dropped


def test_heuristic_empty():
    assert heuristic_glosses("") == []
