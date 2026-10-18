from __future__ import annotations

import uuid
from langchain_core.messages import AIMessage, BaseMessage
from helpers.message_formatter import final_answer

_HANDOFF_PREFIX = "transfer_to_"
_REAL_TOOLS = {"brightmind_knowledge", "web_search", "tavily_search"}


def extract_trace(messages: list[BaseMessage]) -> dict:
    """Derive route + tools_used from a message list.

    Works for the supervisor (routing via 'transfer_to_*' tool calls) and is
    tolerant of a StateGraph that sets AIMessage.name to the node name.
    """
    route: list[str] = []
    tools_used: list[str] = []

    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            name = tc.get("name", "")
            if name.startswith(_HANDOFF_PREFIX):
                specialist = name[len(_HANDOFF_PREFIX):]
                if not route or route[-1] != specialist:
                    route.append(specialist)
            elif name in _REAL_TOOLS:
                tools_used.append(name)

    # For StateGraph nodes, derive route from AIMessage.name values
    if not route:
        for m in messages:
            if isinstance(m, AIMessage):
                name = getattr(m, "name", None)
                if name and name not in ("supervisor", "orchestrator"):
                    if not route or route[-1] != name:
                        route.append(name)

    return {
        "answer": final_answer(messages),
        "route": route,
        "tools_used": tools_used,
        "messages": len(messages),
    }


def make_target(agent):
    """Return a target(inputs) -> trace-dict callable for evaluate()."""

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        thread_id = f"eval-{uuid.uuid4().hex[:8]}"
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return extract_trace(result["messages"])

    return target
