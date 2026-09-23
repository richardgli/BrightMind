from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

def get_web_search_tool():
    return TavilySearch(
        max_results=3,
        topic="general",
    )