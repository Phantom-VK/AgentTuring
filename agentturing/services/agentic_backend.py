"""Agentic backend built on OpenAI Agents SDK and DeepSeek chat completions."""

# pylint: disable=import-outside-toplevel

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generator

from agentturing.config import Settings
from agentturing.constants import TAVILY_DOMAINS
from agentturing.guardrails.setup import make_input_guard, make_output_guard
from agentturing.utils.sanitize_output import format_tavily_results

from .base import BackendResponse


class AgenticBackendUnavailable(RuntimeError):
    """Raised when agentic dependencies or provider settings are unavailable."""


@dataclass
class RuntimeBundle:
    """Runtime objects needed to execute agent routing and research."""

    runner: Any
    triage_agent: Any
    research_agent: Any
    run_config: Any


@lru_cache(maxsize=1)
def _get_vectorstore():
    """Lazily load the vector store to avoid heavy startup imports."""
    from agentturing.database.vectorstore import get_vectorstore

    return get_vectorstore()


@lru_cache(maxsize=8)
def _get_tavily_client(api_key: str):
    """Create and cache the Tavily client used by the research tool."""
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=5,
        topic="general",
        tavily_api_key=api_key,
        include_domains=TAVILY_DOMAINS,
        search_depth="basic",
    )


@lru_cache(maxsize=4)
def _get_openai_client(api_key: str, base_url: str):
    """Create and cache the OpenAI-compatible client for DeepSeek requests."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


class AgenticMathBackend:
    """Backend implementation that routes, researches, and solves math questions."""

    backend_name = "agentic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._input_guard = make_input_guard()
        self._output_guard = make_output_guard()
        self._runtime = self._build_runtime()

    def _build_runtime(self) -> RuntimeBundle:  # pylint: disable=too-many-locals
        """Build the agent routing runtime and its tool-backed research agent."""
        if not self.settings.agentic_enabled:
            raise AgenticBackendUnavailable(
                "Agentic backend requires DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL."
            )

        try:
            from agents import (
                Agent,
                ModelSettings,
                OpenAIProvider,
                RunConfig,
                Runner,
                function_tool,
                set_tracing_disabled,
            )
        except ImportError as exc:
            raise AgenticBackendUnavailable(
                "Agentic backend dependencies are missing. "
                "Run `uv sync` to install pyproject dependencies."
            ) from exc

        set_tracing_disabled(self.settings.tracing_disabled)

        @function_tool
        def search_knowledge_base(query: str, top_k: int = 4) -> str:
            """Search the local math knowledge base for worked examples.

            Args:
                query: The math query or concept to search for.
                top_k: Number of candidate matches to fetch from Qdrant.
            """
            vectorstore = _get_vectorstore()
            results = vectorstore.similarity_search_with_score(
                query,
                k=max(1, min(top_k, 8)),
            )

            if not results:
                return "No relevant knowledge base entries were found."

            formatted_results = []
            for index, (doc, score) in enumerate(results, start=1):
                snippet = " ".join(doc.page_content.split())
                snippet = snippet[:1200]
                formatted_results.append(f"[KB {index}] score={score}\n{snippet}")

            return "\n\n".join(formatted_results)

        @function_tool
        def web_search(query: str) -> str:
            """Search curated math-oriented web sources when local retrieval is insufficient.

            Args:
                query: The math question or research query to search for.
            """
            if not self.settings.tavily_api_key:
                return "Web search is unavailable because TAVILY_API_KEY is not configured."

            tavily = _get_tavily_client(self.settings.tavily_api_key)
            results = tavily.invoke(query)
            formatted = format_tavily_results(results.get("results", []))
            return "\n\n".join(formatted)

        model_settings = ModelSettings(temperature=self.settings.agent_temperature)

        research_agent = Agent(
            name="ResearchAgent",
            handoff_description=(
                "Uses retrieval and web search tools to gather context "
                "for the final DeepSeek reasoner."
            ),
            instructions=(
                "You are a math research specialist. Use the available tools to gather "
                "the most relevant context for the user's question. Return a concise "
                "research brief with only the facts, examples, and retrieved snippets "
                "that will help a separate math solver answer correctly. "
                "Do not produce the final answer unless the user only asked for factual lookup."
            ),
            model=self.settings.research_model,
            model_settings=model_settings,
            tools=[search_knowledge_base, web_search],
        )

        triage_agent = Agent(
            name="TriageAgent",
            instructions=(
                "You route mathematical requests. "
                "If the problem can be answered directly without retrieval "
                "or external context, "
                "respond exactly with NO_RESEARCH_NEEDED. "
                "If it needs retrieval, examples, or external reference material, "
                "handoff to ResearchAgent. "
                "Do not answer non-mathematical requests."
            ),
            model=self.settings.triage_model,
            model_settings=model_settings,
            handoffs=[research_agent],
        )

        provider = OpenAIProvider(
            api_key=self.settings.provider_api_key,
            base_url=self.settings.provider_base_url,
            use_responses=False,
        )
        run_config = RunConfig(
            model_provider=provider,
            model=self.settings.default_model,
        )

        return RuntimeBundle(
            runner=Runner,
            triage_agent=triage_agent,
            research_agent=research_agent,
            run_config=run_config,
        )

    def _gather_context(self, question: str) -> tuple[str, str | None]:
        """Run the routing layer and return either context text or an empty string."""
        result = self._runtime.runner.run_sync(
            self._runtime.triage_agent,
            question,
            run_config=self._runtime.run_config,
        )
        output = str(result.final_output).strip()
        last_agent = getattr(result, "last_agent", None)
        last_agent_name = getattr(last_agent, "name", None)

        if output == "NO_RESEARCH_NEEDED":
            return "", last_agent_name

        return output, last_agent_name

    def _build_reasoner_messages(self, question: str, context: str) -> list[dict[str, str]]:
        """Build the final solver prompt for the DeepSeek reasoner model."""
        system_prompt = (
            "You are an expert mathematics tutor. "
            "Solve the user's problem carefully and provide a clear "
            "step-by-step explanation. "
            "If context is provided, use it only when relevant. "
            "End with `Final Answer:` followed by the result."
        )
        if context:
            user_prompt = f"Question:\n{question}\n\nHelpful context:\n{context}"
        else:
            user_prompt = question
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _stream_reasoner(
        self,
        question: str,
        context: str,
    ) -> Generator[dict[str, str], None, None]:
        """Stream reasoning and answer tokens from the DeepSeek reasoner model."""
        client = _get_openai_client(
            api_key=self.settings.provider_api_key or "",
            base_url=self.settings.provider_base_url or "",
        )
        messages = self._build_reasoner_messages(question, context)
        stream = client.chat.completions.create(
            model=self.settings.solver_model,
            messages=messages,
            temperature=self.settings.agent_temperature,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning_text = getattr(delta, "reasoning_content", None)
            if reasoning_text:
                yield {"type": "reason", "text": reasoning_text}
                continue

            answer_text = getattr(delta, "content", None)
            if answer_text:
                yield {"type": "answer", "text": answer_text}

    def _ask_sync(self, question: str) -> BackendResponse:
        """Resolve a complete answer by routing first, then streaming the final solve."""
        validated_question = self._input_guard(question)
        context, last_agent_name = self._gather_context(validated_question)

        reasoning_chunks: list[str] = []
        answer_chunks: list[str] = []
        for event in self._stream_reasoner(validated_question, context):
            if event["type"] == "reason":
                reasoning_chunks.append(event["text"])
            elif event["type"] == "answer":
                answer_chunks.append(event["text"])

        answer = self._output_guard("".join(answer_chunks).strip())
        reasoning = "".join(reasoning_chunks).strip()

        return BackendResponse(
            answer=answer,
            reasoning=reasoning,
            backend=self.backend_name,
            metadata={
                "last_agent": last_agent_name,
                "model": self.settings.solver_model,
                "research_used": bool(context),
            },
        )

    async def ask(self, question: str) -> BackendResponse:
        """Run the blocking solve flow in a worker thread for FastAPI."""
        return await asyncio.to_thread(self._ask_sync, question)

    def stream_ask(self, question: str) -> Generator[dict[str, Any], None, None]:
        """Yield SSE events without blocking on the research phase first.

        The synchronous `/ask` path keeps the full route -> research -> solve flow.
        For streaming, we prefer immediate token delivery, so this path sends the
        prompt straight to the reasoner and forwards deltas as soon as they arrive.
        """
        validated_question = self._input_guard(question)
        reasoning_chunks: list[str] = []
        answer_chunks: list[str] = []

        yield {
            "type": "meta",
            "backend": self.backend_name,
            "model": self.settings.solver_model,
            "research_used": False,
            "last_agent": "SolverAgent",
        }

        for event in self._stream_reasoner(validated_question, context=""):
            if event["type"] == "reason":
                reasoning_chunks.append(event["text"])
            elif event["type"] == "answer":
                answer_chunks.append(event["text"])
            yield event

        final_answer = self._output_guard("".join(answer_chunks).strip())
        final_reasoning = "".join(reasoning_chunks).strip()
        yield {
            "type": "done",
            "answer": final_answer,
            "reasoning": final_reasoning,
            "metadata": {
                "last_agent": "SolverAgent",
                "model": self.settings.solver_model,
                "research_used": False,
            },
        }
