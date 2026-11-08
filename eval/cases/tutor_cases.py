from __future__ import annotations

DESCRIPTION = (
    "BrightMind multi-turn Socratic tutoring eval. Verifies the tutor keeps "
    "guiding across follow-up turns and does not leak the answer prematurely. "
    "Source of truth lives in eval/cases/tutor_cases.py."
)

CASES: list[dict] = [
    # --- Follow-ups: agent must keep tutoring across multiple turns ---
    {
        "turns": [
            "Help me solve: what's the worst-case height of a BST with n nodes?",
            "Is it log n?",
            "So if it's unbalanced?",
        ],
        "expect_tutor_each_turn": True,
        "expect_no_answer_leak": True,
    },
]
