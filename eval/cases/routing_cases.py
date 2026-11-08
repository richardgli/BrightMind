from __future__ import annotations

DESCRIPTION = (
    "BrightMind single-turn routing + quality eval. Routing order, retrieval "
    "grounding, output shape, and Socratic entry. Source of truth lives in "
    "eval/cases/routing_cases.py."
)

CASES: list[dict] = [
    # --- Routing: quiz requests must research first, then quiz ---
    {
        "question": "Make me a 3-question quiz on binary search trees.",
        "expected_route": ["research", "quiz"],
        "require_retrieval": True,
        "expect_quiz": True,
        "forbid_route": ["tutor"],
        "expect_socratic": False,
    },
    {
        "question": "Quiz me on the attention mechanism in transformers.",
        "expected_route": ["research", "quiz"],
        "require_retrieval": True,
        "expect_quiz": True,
        "forbid_route": ["tutor"],
        "expect_socratic": False,
    },
    # --- Routing: explanation requests should research, not quiz ---
    {
        "question": "Explain how a hash table handles collisions.",
        "expected_route": ["research"],
        "require_retrieval": True,
        "expect_quiz": False,
        "forbid_route": ["quiz", "tutor"],
        "expect_socratic": False,
    },
    {
        "question": "Summarize the key idea of self-attention.",
        "expected_route": ["research"],
        "require_retrieval": True,
        "expect_quiz": False,
        "forbid_route": ["quiz", "tutor"],
        "expect_socratic": False,
    },
    # --- Output shape: a quiz must contain questions ---
    {
        "question": "Give me 2 practice questions about pointers in C.",
        "expected_route": ["research", "quiz"],
        "require_retrieval": True,
        "expect_quiz": True,
        "forbid_route": ["tutor"],
        "expect_socratic": False,
    },
    # --- Tutoring: explicitly asking to be guided and not handing the answer ---
    {
        "question": "I don't get why my binary search returns -1. Don't give me the answer, help me figure it out.",
        "expected_route": ["research", "tutor"],
        "require_retrieval": True,
        "expect_quiz": False,
        "forbid_route": ["quiz"],
        "expect_socratic": True,
    },
]
