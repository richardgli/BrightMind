from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    plan: list[str]
    next: Literal["research", "quiz", "FINISH"]
    research_material: str
    quiz: str
    step_count: int

def main():
    from langgraph.graph import StateGraph, START, END
    def hello(state):
        return {"messages": [("ai", "hi")]}

    g = StateGraph(State)
    g.add_node("hello", hello)
    g.add_edge(START, "hello")
    g.add_edge("hello", END)
    app = g.compile()
    print(app.invoke({"messages": [("user", "text")]}))

if __name__ == "__main__":
    main()