
def format_tavily_results(results: list, min_score: float = 0.75) -> str:
    """Convert Tavily results into a readable context string."""
    formatted = []
    for r in results:
        if r.get("score", 0) >= min_score:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "").strip()
            snippet = content[:500] + "..." if len(content) > 500 else content
            formatted.append(f"### {title}\n{snippet}\n(Source: {url})")
    return formatted if formatted else "No relevant Tavily results found."

