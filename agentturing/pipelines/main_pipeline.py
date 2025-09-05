import logging
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from agentturing.constants import TAVILY_DOMAINS
from agentturing.database.schemas import State
from agentturing.database.vectorstore import get_vectorstore
from agentturing.model.llm import get_llm
from agentturing.prompts import SYSTEM_PROMPT
from agentturing.utils.sanitize_output import format_tavily_results

load_dotenv()

vectorstore = get_vectorstore()
llm_pipeline, tokenizer = get_llm()


tavily = TavilySearch(
    max_results=5,
    topic="general",
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    include_domains = TAVILY_DOMAINS
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

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = llm_pipeline(
        formatted_prompt,
        max_new_tokens=512,
        temperature=0.01,
        top_k=1,
        top_p=1.0,
        do_sample=False,
        repetition_penalty=1.12,
    )

    generated_text = response[0]['generated_text'].strip()
    return generated_text


def retrieve_from_kb(state: State):
    """Function to search knowledge base for relevant context"""
    logging.info("Searching knowledge base for relevant context")

    results = vectorstore.similarity_search_with_score(state["question"], k=6)

    filtered_docs = []
    seen_content = set()
    similarity_threshold = 0.7

    for doc, score in results:
        # Create a snippet for duplicate detection
        snippet = doc.page_content[:200].strip()

        # Skip duplicates
        if snippet in seen_content:
            continue

        # Only include docs with good similarity scores
        if score >= similarity_threshold:
            filtered_docs.append(doc.page_content)
            seen_content.add(snippet)


    logging.info(f"Retrieved {len(filtered_docs)} relevant documents from knowledge base")
    return {"context": filtered_docs}


def evaluate_context_sufficiency(state: State):
    """Function to evaluate if the retrieved context is sufficient to answer the question"""
    context = state.get("context", [])
    question = state["question"]

    # Check if we have any context at all
    if not context or len(context) == 0:
        logging.info("No context found in knowledge base - routing to web search")
        return {"next_step": "web_search"}

    # Check context quality and relevance
    combined_context = "\n".join(context)
    context_length = len(combined_context.strip())

    if context_length < 50:
        logging.info("Context too short - routing to web search")
        return {"next_step": "web_search"}

    # Check if context seems relevant
    question_words = set(question.lower().split())
    context_words = set(combined_context.lower().split())
    overlap = len(question_words.intersection(context_words))

    if overlap < 2:  # Minimal overlap between question and context
        logging.info("Context not sufficiently relevant - routing to web search")
        return {"next_step": "web_search"}

    logging.info("Knowledge base context is sufficient - proceeding to generate answer")
    return {"next_step": "generate_answer"}


def web_search(state: State):
    """Perform web search when knowledge base doesn't have sufficient context"""
    logging.info("Performing web search using Tavily")

    try:
        # Perform web search
        search_results = tavily.invoke(state["question"])

        if search_results and 'results' in search_results:
            # Format web search results - this now returns List[str]
            formatted_results = format_tavily_results(search_results['results'])

            # Combine existing context (if any) with web search results
            existing_context = state.get("context", [])
            updated_context = existing_context + formatted_results

            logging.info(f"Web search completed - added {len(formatted_results)} results")

            # Debug logging to see what we're returning
            logging.info(f"Updated context type: {type(updated_context)}")
            logging.info(f"First context entry type: {type(updated_context[0]) if updated_context else 'No context'}")

            return {"context": updated_context}
        else:
            logging.warning("Web search returned no results")
            return {"context": state.get("context", [])}

    except Exception as e:
        logging.error(f"Web search failed: {str(e)}")
        # Return existing context if web search fails
        return {"context": state.get("context", [])}


def generate_final_answer(state: State):
    """Fuunction to Generate the final answer using available context"""
    logging.info("Generating final answer with available context")

    context = state.get("context", [])
    question = state["question"]

    if context:
        try:
            # Ensure all context items are strings
            string_context = []
            for item in context:
                if isinstance(item, str):
                    string_context.append(item)
                else:
                    string_context.append(str(item))

            # Combine all context into a single string
            combined_context = "\n\n".join(string_context)

            logging.info(f"Combined context length: {len(combined_context)}")
            response = generate_with_llm(question, combined_context)

        except Exception as e:
            logging.error(f"Error combining context: {str(e)}")
            response = generate_with_llm(question)
    else:

        logging.warning("No context available - generating answer without context")
        response = generate_with_llm(question)

    return {"answer": response}


def route_after_evaluation(state: State):
    """Router function to determine next step after context evaluation"""
    return state.get("next_step", "generate_answer")


def build_graph():
    """Build and compile the state graph for the RAG + MCP pipeline"""

    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("retrieve_kb", retrieve_from_kb)
    graph_builder.add_node("evaluate_context", evaluate_context_sufficiency)
    graph_builder.add_node("web_search", web_search)
    graph_builder.add_node("generate_answer", generate_final_answer)

    # Define the flow
    graph_builder.add_edge(START, "retrieve_kb")
    graph_builder.add_edge("retrieve_kb", "evaluate_context")

    # Conditional routing after context evaluation
    graph_builder.add_conditional_edges(
        "evaluate_context",
        route_after_evaluation,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer"
        }
    )

    # After web search, go directly to answer generation
    graph_builder.add_edge("web_search", "generate_answer")

    # End after generating answer
    graph_builder.add_edge("generate_answer", END)

    return graph_builder.compile()
