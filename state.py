from typing import Annotated, Literal
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
