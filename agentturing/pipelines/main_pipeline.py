import logging
import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from agentturing.database.schemas import State
from agentturing.database.vectorstore import get_vectorstore
from agentturing.model.llm import get_llm
from agentturing.prompts import SYSTEM_PROMPT
from agentturing.utils.sanitize_output import format_tavily_results

load_dotenv()

vectorstore = get_vectorstore()
llm_pipeline, tokenizer = get_llm()

tavily = TavilySearch(
    max_results=3,
    topic="general",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/main_pipeline.log",
    filemode="w"
)


def generate_with_llm(question, context=""):
    """Helper function to generate responses using the chat template"""
    # Combine system prompt with context if available
    system_content = SYSTEM_PROMPT
    if context:
        system_content += "\n\nContext:\n" + context

    # Format messages using chat template for Qwen
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate response with generation parameters
    response = llm_pipeline(
        formatted_prompt,
        max_new_tokens=400,
        temperature=0.01,
        top_k=1,
        top_p=1.0,
        do_sample=False,
        repetition_penalty=1.12,
    )

    generated_text = response[0]['generated_text'].strip()

    # Post-process to enforce system prompt rules
    return generated_text


def generate_initial(state: State):
    logging.info("Generating initial answer without context")
    response = generate_with_llm(state["question"])
    return {"answer": response}


def check_initial(state: State):
    answer = state.get("answer", "")
    if "I don't know" in answer or "Let's search the web" in answer or "not provide" in answer or "not provide" in answer or "I can't assist" in answer:
        return {"next_step": "retrieve"}
    return {"next_step": "end"}


def retrieve(state: State):
    logging.info("Searching knowledge base")
    results = vectorstore.similarity_search_with_score(state["question"])
    filtered = []
    seen = set()
    max_docs = 4
    for doc, score in results:
        snippet = doc.page_content[:100]
        if snippet in seen:
            continue
        if score >= 0.8:
            filtered.append(doc)
            seen.add(snippet)
        if len(filtered) >= max_docs:
            break
    return {"context": [doc.page_content for doc in filtered]}


def tavily_search(state: State):
    logging.info("Fallback to Tavily web search")
    results = tavily.invoke(state["question"])
    formatted_results = format_tavily_results(results['results'])
    return {"context": formatted_results}


def generate(state: State):
    logging.info("Generating answer with context")
    docs_content = "\n\n".join(state.get("context", []))
    response = generate_with_llm(state["question"], docs_content)
    return {"answer": response}



def route_after_initial(state: State):
    return state.get("next_step", "end")

def route_after_retrieve(state: State):
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
        route_after_initial,
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