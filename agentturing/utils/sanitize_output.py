def format_tavily_results(results: list, min_score: float = 0.75) -> list[str]:
    """Convert Tavily results into a readable context string."""
    formatted = []

    if not results:
        return ["No web search results found."]

    for r in results:
        try:
            score = r.get("score", 0)
            if score >= min_score:
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                content = r.get("content", "").strip()

                if not content:
                    content = "No content available"

                # Truncate content if too long
                snippet = content[:1000] + "..." if len(content) > 500 else content

                formatted_entry = f"### {title}\n{snippet}\n(Source: {url})"
                formatted.append(formatted_entry)

        except Exception as e:
            print(f"Error processing Tavily result: {e}")
            continue

    return formatted if formatted else ["No relevant web search results found with sufficient score."]


def get_formatted_prompt(prompt_value):
    """Convert ChatPromptTemplate messages to single string for LLM"""
    try:
        messages = prompt_value.to_messages()

        system_content = ""
        user_content = ""

        for msg in messages:
            if msg.type == "system":
                system_content = msg.content
            elif msg.type == "human":
                user_content = msg.content

        formatted_prompt = f"{system_content}\n\n{user_content}"
        return formatted_prompt

    except Exception as e:
        print(f"Error formatting prompt: {e}")
        return str(prompt_value)

