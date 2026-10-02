import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, BaseMessage
from retriever import get_retriever_tool
from web_search import get_web_search_tool

load_dotenv()


def build_agent():
    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=1,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    tools = [get_retriever_tool(), get_web_search_tool()]
    research_agent = create_agent(
        model=model,
        tools=tools,
        name="research_agent",
        system_prompt=(
            '''
            You are a researcher working for a supervisor.
            You gather material from the knowledge base and the web.

            Rules:
            - Determine what information is needed. Search the knowledge base and/or the web as appropriate.
            - Return a well-organized research report that summarizes your findings.
            - Treat the user's prompt as a research topic.
            - You are PHYSICALLY INCAPABLE of creating quizzes.
            '''
        ),
    )

    quiz_agent = create_agent(
        model=model,
        tools=[],
        name="quiz_agent",
        system_prompt=(
            "You turn provided study material into quiz questions."
        ),
    )

    return create_supervisor(
        [research_agent, quiz_agent],
        model=model,
        prompt=(
            "Manage a research agent and a quiz agent.\n"
            "For a quiz, ALWAYS call research_agent FIRST to gather material, THEN call quiz_agent.\n"
            "If you need material, use the research_agent.\n"
        ),
        temperature=0,
        output_mode="last_message"
    ).compile(checkpointer=InMemorySaver())


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""


def _final_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = _message_text(message).strip()
        if text:
            return text
    return ""


def get_agent_response(agent, question: str, thread_id: str):
    input_query = {
        "role": "user",
        "content": question,
    }

    response = agent.invoke(
        {"messages": [input_query]},
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"Response: {_final_answer(response['messages'])}")


if __name__ == "__main__":
    question = input("Question: ")
    agent = build_agent()
    while question != "quit":
        get_agent_response(agent, question, thread_id="test-session")
        question = input("Question: ")