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


def _question_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _is_question_line(line)]


def _count_questions(text: str) -> int:
    """Count genuine quiz questions."""
    interrogative_lines = len(_question_lines(text))
    labeled = len(_QLABEL_RE.findall(text))
    return max(interrogative_lines, labeled)


def quiz_shape(run, example) -> dict:
    expect = _reference(example).get("expect_quiz", False)
    output = _outputs(run)

    if output.get("structured"):
        quiz = output.get("quiz")
        n = len(quiz.get("questions", [])) if quiz else 0
        source = "structured"
    else:
        n = _count_questions(output.get("answer", "") or "")
        source = "text"

    threshold = 1 if source == "structured" else _QUIZ_MIN_QUESTIONS
    is_quiz = n >= threshold
    passed = is_quiz if expect else not is_quiz
    return {
        "key": "quiz_shape",
        "score": int(passed),
        "comment": f"questions={n} source={source} threshold={threshold}",
    }


def socratic_behaviour(run, example) -> dict:
    """Judge: when the student is wrong, does the tutor guide (ask a question) rather than reveal the answer?"""
    from langchain_anthropic import ChatAnthropic

    expect = _reference(example).get("expect_socratic", False)
    if not expect:
        return {"key": "socratic_behaviour", "score": None,
                "comment": "not a tutoring case"}

    question = (example.inputs or {}).get("question", "") if example else ""
    answer = _outputs(run).get("answer", "") or ""
    if not answer:
        return {"key": "socratic_behaviour", "score": 0, "comment": "empty answer"}

    judge = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL"),
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    prompt = (
        "You are grading a Socratic tutor. The student gave a wrong/incomplete answer. "
        "Score 1 ONLY IF if the tutor does NOT reveal the correct answer, and "
        "asks at least one guiding question. Score 0 if the tutor states the answer outright. "
        "Reply with ONLY '1' or '0', then a space, then a brief reason.\n\n"
        f"QUESTION:\n{question}\n\nREPLY:\n{answer[:2000]}"
    )
    resp = judge.invoke([{"role": "user", "content": prompt}])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = text.strip()
    score = 1 if text.startswith("1") else 0
    return {"key": "socratic_behaviour", "score": score, "comment": text[:200]}


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


def tutor_each_turn(run, example) -> dict:
    """Multi-turn: every follow-up turn should route through the tutor node.

    `routes` is a list-of-lists: one route per user turn. The first turn may
    include research (grounding), so we require 'tutor' in every follow-up
    turn (turns after the first).
    """
    expect = _reference(example).get("expect_tutor_each_turn", False)
    if not expect:
        return {"key": "tutor_each_turn", "score": None,
                "comment": "not a multi-turn tutoring case"}

    routes = _outputs(run).get("routes", [])
    if not routes:
        return {"key": "tutor_each_turn", "score": 0, "comment": "no routes captured"}

    followups = routes[1:] if len(routes) > 1 else []
    if not followups:
        return {"key": "tutor_each_turn", "score": 0,
                "comment": f"no follow-up turns; routes={routes}"}

    passed = all("tutor" in r for r in followups)
    return {
        "key": "tutor_each_turn",
        "score": int(passed),
        "comment": f"routes={routes}",
    }


def no_answer_leak(run, example) -> dict:
    """Judge: across the tutoring turns before the final one, the tutor must
    not hand over the final answer outright."""
    from langchain_anthropic import ChatAnthropic

    expect = _reference(example).get("expect_no_answer_leak", False)
    if not expect:
        return {"key": "no_answer_leak", "score": None,
                "comment": "not a multi-turn tutoring case"}

    replies = _outputs(run).get("replies", [])
    if len(replies) < 2:
        return {"key": "no_answer_leak", "score": 0,
                "comment": f"need >=2 replies; got {len(replies)}"}

    # Only the replies before the last turn must withhold the answer.
    early = "\n\n---\n\n".join(replies[:-1])

    judge = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL"),
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    prompt = (
        "You are grading a Socratic tutor across its early replies (before the "
        "student has reached the answer). Score 1 if the tutor GUIDES with "
        "questions and does NOT state the final answer outright in these early "
        "replies. Score 0 if it reveals the answer prematurely. Reply with ONLY "
        "'1' or '0', then a space, then a brief reason.\n\n"
        f"EARLY TUTOR REPLIES:\n{early[:3000]}"
    )
    resp = judge.invoke([{"role": "user", "content": prompt}])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = text.strip()
    score = 1 if text.startswith("1") else 0
    return {"key": "no_answer_leak", "score": score, "comment": text[:200]}


DETERMINISTIC_EVALUATORS = [
    routing_order,
    no_forbidden_route,
    retrieval_grounding,
    quiz_shape,
]

JUDGE_EVALUATORS = [faithfulness, socratic_behaviour]

ALL_EVALUATORS = DETERMINISTIC_EVALUATORS + JUDGE_EVALUATORS

# Multi-turn tutoring suite evaluators (used by run.py --suite tutor).
MULTITURN_EVALUATORS = [tutor_each_turn, no_answer_leak]
