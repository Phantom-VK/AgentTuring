
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


def get_formatted_prompt(prompt_value):
    """Convert ChatPromptTemplate messages to single string for LLM"""
    messages = prompt_value.to_messages()

    # Extract system and user content
    system_content = ""
    user_content = ""

    for msg in messages:
        if msg.type == "system":
            system_content = msg.content
        elif msg.type == "human":
            user_content = msg.content

    # Format as single prompt string
    formatted_prompt = f"{system_content}\n\n{user_content}"
    return formatted_prompt

