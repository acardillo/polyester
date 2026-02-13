"""
Small classifier for structural query intent. No manual training: uses baked-in
examples and trains at first use (optionally cached to disk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# Baked-in training data: (query, label). Labels map to (edge_type, use_successors).
# "semantic" -> not structural. Others -> structural with that edge/direction.
STRUCTURAL_INTENT_EXAMPLES = [
    # semantic (not structural)
    ("How do I parse JSON from a string?", "semantic"),
    ("Serialize Python object to JSON", "semantic"),
    ("Work with file paths and directories", "semantic"),
    ("Regular expression match and search", "semantic"),
    ("Split string by regex pattern", "semantic"),
    ("Read and write CSV files", "semantic"),
    ("Default dictionary with default value", "semantic"),
    ("Partial function application", "semantic"),
    ("Encode and decode base64", "semantic"),
    ("Thread-safe queue", "semantic"),
    ("Abstract base class and metaclass", "semantic"),
    ("Deep copy and shallow copy", "semantic"),
    ("Context manager and with statement", "semantic"),
    ("How do I use inheritance in Python?", "semantic"),
    ("Explain callbacks and async", "semantic"),
    ("What is the difference between list and tuple?", "semantic"),
    ("re.compile", "semantic"),
    ("base64.b64encode", "semantic"),
    # structural: what does X call? (successors, calls)
    ("What does json.load call internally?", "structural_successors_calls"),
    ("What does json.load call?", "structural_successors_calls"),
    ("Which function does json.load invoke?", "structural_successors_calls"),
    ("What does the JSON function that deserializes a file-like object call internally?", "structural_successors_calls"),
    ("What does X call?", "structural_successors_calls"),
    ("What does pathlib.Path call?", "structural_successors_calls"),
    # structural: what calls X? (predecessors, calls)
    ("What functions call json.loads?", "structural_predecessors_calls"),
    ("What calls json.loads?", "structural_predecessors_calls"),
    ("Who calls json.loads?", "structural_predecessors_calls"),
    ("Which function calls the one that deserializes a string containing a JSON document?", "structural_predecessors_calls"),
    ("What functions call this?", "structural_predecessors_calls"),
    ("Callers of json.loads", "structural_predecessors_calls"),
    # structural: inheritance (successors, base_class)
    ("What classes does pathlib.Path inherit from?", "structural_successors_base_class"),
    ("What does Path inherit from?", "structural_successors_base_class"),
    ("What is the base class of pathlib.Path?", "structural_successors_base_class"),
    ("What is the base class of the pathlib class that can make system calls on path objects?", "structural_successors_base_class"),
    ("What does pathlib.Path inherit from?", "structural_successors_base_class"),
]

LABEL_TO_INTENT = {
    "semantic": (False, None, True),
    "structural_successors_calls": (True, "calls", True),
    "structural_predecessors_calls": (True, "calls", False),
    "structural_successors_base_class": (True, "base_class", True),
}

_classifier: Optional["_StructuralIntentClassifier"] = None


def classify(query_text: str) -> Optional[Tuple[bool, Optional[str], bool]]:
    """
    Classify query intent. Returns (is_structural, edge_type_or_none, use_successors),
    or None if sklearn is not available (caller should use keyword fallback).
    """
    global _classifier
    if _classifier is None:
        _classifier = _StructuralIntentClassifier()
    return _classifier.predict(query_text)


class _StructuralIntentClassifier:
    """Train a small LogisticRegression on baked-in examples; predict at runtime."""

    def __init__(self) -> None:
        self._model = None
        self._vectorizer = None
        self._fit()

    def _fit(self) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return
        queries = [q for q, _ in STRUCTURAL_INTENT_EXAMPLES]
        labels = [lbl for _, lbl in STRUCTURAL_INTENT_EXAMPLES]
        self._vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
            lowercase=True,
        )
        X = self._vectorizer.fit_transform(queries)
        self._model = LogisticRegression(max_iter=500, random_state=42)
        self._model.fit(X, labels)

    def predict(self, query_text: str) -> Optional[Tuple[bool, Optional[str], bool]]:
        if self._model is None or self._vectorizer is None:
            return None
        X = self._vectorizer.transform([query_text])
        label = self._model.predict(X)[0]
        return LABEL_TO_INTENT[label]
