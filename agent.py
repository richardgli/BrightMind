import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
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

    agent = create_agent(model, tools=tools)

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