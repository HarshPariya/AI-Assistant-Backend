import json
from ddgs import DDGS

def search_web(query: str) -> str:
    """
    Search the web for real-time information using DuckDuckGo.
    Returns a formatted string of the top 3-5 results.
    """
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No web search results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
        
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Error during web search: {str(e)}"
