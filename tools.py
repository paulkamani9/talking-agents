# tools.py

import logging

logging.getLogger("ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)

try:
    from ddgs import DDGS
except ModuleNotFoundError:
    try:
        from duckduckgo_search import DDGS
    except ModuleNotFoundError:
        DDGS = None

def search_web(query, max_results=3):
    """
    Search the web using DuckDuckGo.
    Returns a formatted string of results.
    """
    
    if DDGS is None:
        return "Search package not installed. Install with: pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, backend="lite"))
    except Exception as e:
        return f"Search failed: {e}"
    
    if not results:
        return "No results found."
    
    output = ""
    
    for r in results:
        title = r.get("title", "No title")
        body = r.get("body", "")
        output += f"- {title}: {body}\n"
        
    return output