import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from retriever import get_retriever_tool
from web_search import get_web_search_tool

load_dotenv()

def get_agent_response():
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
        system_prompt="You gather material from the knowledge base and the web. DO NOT write quizzes; only gather and summarize information."
    )

    quiz_agent = create_agent(
        model=model,
        tools=[],
        name="quiz_agent",
        system_prompt="You turn provided study material into quiz questions."
    )

    orchestrator_agent = create_supervisor(
        [research_agent, quiz_agent],
        model=model,
        prompt=(
            "Manage a research agent and a quiz agent.\n"
            "For a quiz, ALWAYS call research_agent FIRST to gather material, THEN call quiz_agent."
            "Never call quiz_agent before research_agent has returned material."
        )
    ).compile()

    question = input("Question: ")
    input_query = {
        "role": "user",
        "content": question,
    }

    response = agent.invoke({
        "messages": [input_query]
    })

    print(f"Response: {response["messages"][-1].content}")


if __name__ == "__main__":
    get_agent_response()