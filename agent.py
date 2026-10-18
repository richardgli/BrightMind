import os
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from retriever import get_retriever_tool
from web_search import get_web_search_tool
from state import State
from agent_helper import final_answer

load_dotenv()

ORCH_PROMPT = '''
    You are a study assistant.\n
    For a quiz, route to research FIRST, THEN route to quiz.\n
    If you need material, use the research.\n
'''

RESEARCH_PROMPT = '''
    You are a researcher working for a supervisor.
    You gather material from the knowledge base and the web.

    Rules:
    - Determine what information is needed. Always search the knowledge base first; use web search for supplementary context.
    - Return a summary of your findings.
    - Treat the user's prompt as a research topic.
'''

QUIZ_PROMPT = '''
    You turn provided study material into quiz questions.
'''

model = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL"),
    temperature=1,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


class RoutingDecision(BaseModel):
    plan: list[str] = Field(description="subtasks for the goal")
    next: Literal["research", "quiz", "FINISH"]
    reasoning: str


def orchestrator(state):
    if state.get("step_count", 0) > 5:
        return {"next": "FINISH", "plan": [""], "step_count": state.get("step_count", 0) + 1}

    router = model.with_structured_output(RoutingDecision)
    decision = router.invoke([
        {"role": "system", "content": ORCH_PROMPT},
        *state["messages"],
    ])

    return {"next": decision.next, "plan": decision.plan, "step_count": state.get("step_count", 0) + 1}


def research(state):
    tools = [get_retriever_tool(), get_web_search_tool()]
    agent = create_agent(
        model=model,
        tools=tools,
        name="research_agent",
        system_prompt=RESEARCH_PROMPT,
    )

    res = agent.invoke({"messages": state["messages"]})
    material = res["messages"][-1].content
    return {"research_material": material, "messages": [res["messages"][-1]]}


def quiz(state):
    agent = create_agent(
        model=model,
        tools=[],
        name="quiz_agent",
        system_prompt=QUIZ_PROMPT,
    )

    res = agent.invoke({
        "messages": [{"role": "user", "content": state["research_material"]}]
    })
    material = res["messages"][-1].content
    return {"quiz": material, "messages": [res["messages"][-1]]}


def route(state):
    return state["next"]


def get_agent_response(agent, question: str, thread_id: str):
    input_query = {
        "role": "user",
        "content": question,
    }

    response = agent.invoke(
        {"messages": [input_query]},
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"Response: {final_answer(response['messages'])}")


def build_stategraph():
    g = StateGraph(State)

    g.add_node("orchestrator", orchestrator)
    g.add_node("research", research)
    g.add_node("quiz", quiz)

    g.add_edge(START, "orchestrator")
    
    g.add_conditional_edges("orchestrator", route, {
        "research": "research", "quiz": "quiz", "FINISH": END
    })

    for node in ["research", "quiz"]:
        g.add_edge(node, "orchestrator")

    return g.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    question = input("Question: ")
    agent = build_stategraph()
    while question != "quit":
        get_agent_response(agent, question, thread_id="test-session")
        question = input("Question: ")