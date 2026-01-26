"""
Tavily search tool for LangChain agent (compatible with Groq).

Usage:
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(api_key=..., max_results=3)
    # or use get_tavily_tool(api_key)
"""
from langchain_tavily import TavilySearch

def get_tavily_tool(api_key: str, max_results: int = 3):
    return TavilySearch(api_key=api_key, max_results=max_results)
