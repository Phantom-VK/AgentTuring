import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from agentturing.database.schemas import State
from agentturing.database.vectorstore import get_vectorstore
from agentturing.model.llm import get_llm
from agentturing.prompts import SYSTEM_PROMPT
from agentturing.utils.sanitize_output import format_tavily_results

load_dotenv()

vectorstore = get_vectorstore()
llm = get_llm()
tavily = TavilySearch(
    max_results=3,
    topic="general",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

prompt_template = ChatPromptTemplate([
    ("system", SYSTEM_PROMPT + "\n\n{context}\n\n "),
    ("user", "{question}\n\nAnswer-step-by-step:")
])


def generate_initial(state: State):
    # Generate answer without any context
    prompt = prompt_template.invoke({"context": "", "question": state["question"]})
    response = llm.predict(prompt.to_string())
    return {"answer": response}


def check_initial(state: State):
    # Check if the initial answer indicates uncertainty
    answer = state.get("answer", "")
    if "I don't know" in answer or "Let's search the web" in answer:
        return {"next_step": "retrieve"}  # Return dict, not string
    else:
        return {"next_step": "end"}  # Return dict, not string


def retrieve(state: State):
    # Retrieve context from RAG
    print("Retrieving from RAG")
    results = vectorstore.similarity_search_with_score(state["question"])
    filtered_docs = []
    seen_content = set()
    max_results = 4

    for doc, score in results:
        # Skip duplicates or very similar content
        content_hash = hash(doc.page_content[:100])  # First 100 chars as hash
        if content_hash in seen_content:
            continue

        if score >= 0.75:
            filtered_docs.append(doc)
            seen_content.add(content_hash)

        if len(filtered_docs) >= max_results:
            break

    return {"context": [[doc.page_content for doc in filtered_docs]]}


def tavily_search(state: State):
    # Search using Tavily if RAG doesn't have context
    print("Searching web!!")
    results = tavily.invoke(state["question"])
    web_docs = format_tavily_results(results['results'])
    return {"context": web_docs}  # Replace context with web results


def generate(state: State):
    # Generate answer with context
    docs_content = "\n\n".join(state.get("context", []))
    prompt = prompt_template.invoke({"context": docs_content, "question": state["question"]})
    response = llm.predict(prompt.to_string())
    return {"answer": response}


def route_after_initial(state: State):
    # Route based on next_step field
    return state.get("next_step", "end")


def route_after_retrieve(state: State):
    # Decide after retrieval: if no context, go to Tavily; else generate
    if not state.get("context"):
        return "tavily"
    return "generate"


def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("generate_initial", generate_initial)
    graph_builder.add_node("check_initial", check_initial)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("tavily", tavily_search)
    graph_builder.add_node("generate", generate)

    graph_builder.add_edge(START, "generate_initial")
    graph_builder.add_edge("generate_initial", "check_initial")
    graph_builder.add_conditional_edges(
        "check_initial",
        route_after_initial,  # Use routing function instead of direct return
        {"retrieve": "retrieve", "end": END}
    )
    graph_builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"tavily": "tavily", "generate": "generate"}
    )
    graph_builder.add_edge("tavily", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()
