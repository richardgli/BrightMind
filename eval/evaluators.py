from __future__ import annotations

import os
import re


def _outputs(run):
    return run.outputs or {}


def _reference(example):
    return (example.outputs or {}) if example else {}


def routing_order(run, example) -> dict:
    expected = _reference(example).get("expected_route", [])
    actual = _outputs(run).get("route", [])

    it = iter(actual)
    passed = all(any(step == a for a in it) for step in expected)
    return {
        "key": "routing_order",
        "score": int(passed),
        "comment": f"expected is ordered subsequence of actual? expected={expected} actual={actual}",
    }


def no_forbidden_route(run, example) -> dict:
    forbidden = set(_reference(example).get("forbid_route", []))
    actual = set(_outputs(run).get("route", []))
    violated = forbidden & actual
    return {
        "key": "no_forbidden_route",
        "score": int(not violated),
        "comment": f"forbidden={sorted(forbidden)} hit={sorted(violated)}",
    }


def retrieval_grounding(run, example) -> dict:
    require = _reference(example).get("require_retrieval", False)
    tools_used = _outputs(run).get("tools_used", [])
    if not require:
        return {"key": "retrieval_grounding", "score": 1,
                "comment": "retrieval not required"}
    passed = len(tools_used) > 0
    return {
        "key": "retrieval_grounding",
        "score": int(passed),
        "comment": f"required; tools_used={tools_used}",
    }


_QUIZ_MIN_QUESTIONS = 2
# A line that STARTS as a numbered/labeled item: "1.", "2)", "Q3", "Question 4".
_ITEM_START_RE = re.compile(r"^\s*(?:q(?:uestion)?\s*)?\d+\s*[.)]", re.IGNORECASE)
_INTERROGATIVE_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|whose|whom|"
    r"is|are|was|were|do|does|did|can|could|would|should|will|"
    r"name|define|describe|explain|list|state|identify|calculate|"
    r"true or false|which of)\b",
    re.IGNORECASE,
)
_QLABEL_RE = re.compile(r"\bquestion\s*\d|\bq\d\b", re.IGNORECASE)


def _is_question_line(line: str) -> bool:
    """A line counts as a quiz question only if it is actually interrogative."""
    stripped = line.strip()
    if not stripped:
        return False
    # Remove a leading "1." / "2)" / "Q3." / "Question 4:" marker, if present.
    body = _ITEM_START_RE.sub("", stripped).lstrip(" :.-")
    if body.endswith("?"):
        return True
    return bool(_INTERROGATIVE_RE.match(body))


def _count_questions(text: str) -> int:
    """Count genuine quiz questions."""
    lines = text.splitlines()
    interrogative_lines = sum(1 for line in lines if _is_question_line(line))
    labeled = len(_QLABEL_RE.findall(text))
    return max(interrogative_lines, labeled)


def quiz_shape(run, example) -> dict:
    expect = _reference(example).get("expect_quiz", False)
    answer = _outputs(run).get("answer", "") or ""
    n_questions = _count_questions(answer)
    looks_like_quiz = n_questions >= _QUIZ_MIN_QUESTIONS
    if expect:
        passed = looks_like_quiz
        comment = (
            f"expected quiz; detected {n_questions} question(s) "
            f"(need >={_QUIZ_MIN_QUESTIONS})"
        )
    else:
        passed = not looks_like_quiz
        comment = (
            f"expected non-quiz; detected {n_questions} question(s) "
            + ("passed" if passed else "unexpectedly quiz-shaped")
        )
    return {"key": "quiz_shape", "score": int(passed), "comment": comment}


def faithfulness(run, example) -> dict:
    """LLM-as-judge: is the answer coherent and non-fabricated for the question?

    Kept lightweight and self-contained so the harness has no extra deps.
    Returns score in [0,1].
    """
    from langchain_anthropic import ChatAnthropic

    question = (example.inputs or {}).get("question", "") if example else ""
    answer = _outputs(run).get("answer", "") or ""
    if not answer:
        return {"key": "faithfulness", "score": 0, "comment": "empty answer"}

    judge = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL"),
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    prompt = (
        "You are grading an AI study assistant's answer. Score 1 if the answer "
        "is on-topic for the question, internally coherent, and does not contain "
        "obviously fabricated claims; score 0 otherwise. Reply with ONLY '1' or '0', "
        "then a space, then a brief reason.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer[:2000]}"
    )
    resp = judge.invoke([{"role": "user", "content": prompt}])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = text.strip()
    score = 1 if text.startswith("1") else 0
    return {"key": "faithfulness", "score": score, "comment": text[:200]}


DETERMINISTIC_EVALUATORS = [
    routing_order,
    no_forbidden_route,
    retrieval_grounding,
    quiz_shape,
]

JUDGE_EVALUATORS = [faithfulness]

ALL_EVALUATORS = DETERMINISTIC_EVALUATORS + JUDGE_EVALUATORS
