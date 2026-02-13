"""Tests for intent classifiers."""

import pytest

from src.classifiers.structural_intent_classifier import (
    LABEL_TO_INTENT,
    STRUCTURAL_INTENT_EXAMPLES,
    classify,
)


class TestStructuralIntentClassifierData:
    """Sanity checks on baked-in training data."""

    def test_every_example_label_has_intent_mapping(self):
        """Every label in STRUCTURAL_INTENT_EXAMPLES must exist in LABEL_TO_INTENT."""
        for _query, label in STRUCTURAL_INTENT_EXAMPLES:
            assert label in LABEL_TO_INTENT, f"Missing LABEL_TO_INTENT for {label!r}"

    def test_intent_tuples_are_three_elements(self):
        """Each intent is (is_structural, edge_type_or_none, use_successors)."""
        for label, intent in LABEL_TO_INTENT.items():
            assert isinstance(intent, tuple), f"{label}: intent must be tuple"
            assert len(intent) == 3, f"{label}: intent must have 3 elements"
            is_structural, edge_type, use_successors = intent
            assert isinstance(is_structural, bool)
            assert edge_type is None or isinstance(edge_type, str)
            assert isinstance(use_successors, bool)


class TestStructuralIntentClassify:
    """Tests for classify() with sklearn available (in requirements)."""

    def test_returns_three_element_tuple_or_none(self):
        """classify returns (bool, Optional[str], bool) or None."""
        result = classify("How do I parse JSON?")
        if result is None:
            pytest.skip("sklearn not available")
        assert isinstance(result, tuple)
        assert len(result) == 3
        is_structural, edge_type, use_successors = result
        assert isinstance(is_structural, bool)
        assert edge_type is None or isinstance(edge_type, str)
        assert isinstance(use_successors, bool)

    def test_semantic_query_returns_non_structural(self):
        """Conceptual/semantic queries are classified as not structural."""
        result = classify("How do I parse JSON from a string?")
        if result is None:
            pytest.skip("sklearn not available")
        is_structural, edge_type, use_successors = result
        assert is_structural is False
        assert edge_type is None
        assert use_successors is True

    def test_semantic_like_queries(self):
        """Other semantic-style queries stay non-structural."""
        queries = [
            "Serialize Python object to JSON",
            "Work with file paths and directories",
            "What is the difference between list and tuple?",
        ]
        for query in queries:
            result = classify(query)
            if result is None:
                pytest.skip("sklearn not available")
            is_structural, _, _ = result
            assert is_structural is False, f"Expected semantic for {query!r}"

    def test_what_does_x_call_returns_successors_calls(self):
        """Queries about what X calls → structural, calls, successors."""
        result = classify("What does json.load call internally?")
        if result is None:
            pytest.skip("sklearn not available")
        is_structural, edge_type, use_successors = result
        assert is_structural is True
        assert edge_type == "calls"
        assert use_successors is True

    def test_what_calls_x_returns_predecessors_calls(self):
        """Queries about what calls X → structural, calls, predecessors (use_successors=False)."""
        result = classify("What functions call json.loads?")
        if result is None:
            pytest.skip("sklearn not available")
        is_structural, edge_type, use_successors = result
        assert is_structural is True
        assert edge_type == "calls"
        assert use_successors is False

    def test_inherit_base_class_returns_successors_base_class(self):
        """Queries about inheritance/base class → structural, base_class, successors."""
        result = classify("What does pathlib.Path inherit from?")
        if result is None:
            pytest.skip("sklearn not available")
        is_structural, edge_type, use_successors = result
        assert is_structural is True
        assert edge_type == "base_class"
        assert use_successors is True

    def test_callers_of_returns_predecessors(self):
        """'Callers of X' phrasing → predecessors, calls."""
        result = classify("Callers of json.loads")
        if result is None:
            pytest.skip("sklearn not available")
        is_structural, edge_type, use_successors = result
        assert is_structural is True
        assert edge_type == "calls"
        assert use_successors is False
