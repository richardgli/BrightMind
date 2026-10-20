from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    plan: list[str]
    next: Literal["research", "quiz", "FINISH"]
    research_material: str
    quiz: dict
    step_count: int

class QuizQuestion(BaseModel):
    question: str = Field(description="The question text.")
    choices: list[str] = Field(default_factory=list,
                               description="Answer options (empty for open-ended).")
    answer: str = Field(description="The correct answer.")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Calibrated difficulty.")

class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion] = Field(description="The quiz questions.")
