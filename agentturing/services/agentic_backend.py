"""Agentic backend built on streamed OpenAI Agents SDK orchestration."""

# pylint: disable=import-outside-toplevel

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from agentturing.config import Settings
from agentturing.constants import TAVILY_DOMAINS
from agentturing.guardrails.setup import make_input_guard, make_output_guard
from agentturing.utils.sanitize_output import format_tavily_results

from .base import BackendResponse


class AgenticBackendUnavailable(RuntimeError):
    """Raised when agentic dependencies or provider settings are unavailable."""


@dataclass
class RuntimeBundle:
    """Runtime objects needed to execute the streamed agent workflow."""

    runner: Any
    router_agent: Any
    solver_agent: Any
    math_research_agent: Any
    web_research_agent: Any
    run_config: Any


@lru_cache(maxsize=1)
def _get_vectorstore():
    """Lazily load the vector store to avoid heavy startup imports."""
    from agentturing.database.vectorstore import get_vectorstore

    return get_vectorstore()


@lru_cache(maxsize=8)
def _get_math_tavily_client(api_key: str):
    """Create and cache the Tavily client used for math-focused research."""
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=5,
        topic="general",
        tavily_api_key=api_key,
        include_domains=TAVILY_DOMAINS,
        search_depth="basic",
    )


@lru_cache(maxsize=8)
def _get_general_tavily_client(api_key: str):
    """Create and cache the Tavily client used for general web research."""
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=5,
        topic="general",
        tavily_api_key=api_key,
        search_depth="advanced",
    )


class AgenticMathBackend:
    """Backend implementation that streams one agentic run end to end."""

    backend_name = "agentic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._input_guard = make_input_guard()
        self._output_guard = make_output_guard()
        self._runtime = self._build_runtime()

    def _build_runtime(self) -> RuntimeBundle:  # pylint: disable=too-many-locals
        """Build the streamed agent runtime and its tool-backed research agent."""
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
                set_default_openai_api,
                set_tracing_disabled,
            )
        except ImportError as exc:
            raise AgenticBackendUnavailable(
                "Agentic backend dependencies are missing. "
                "Run `uv sync` to install pyproject dependencies."
            ) from exc

        set_default_openai_api("chat_completions")
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

        def _run_tavily_search(query: str, math_only: bool) -> str:
            """Run Tavily search with the correct source scope for the current path.

            Args:
                query: The math question or research query to search for.
            """
            if not self.settings.tavily_api_key:
                return "Web search is unavailable because TAVILY_API_KEY is not configured."

            try:
                tavily = (
                    _get_math_tavily_client(self.settings.tavily_api_key)
                    if math_only
                    else _get_general_tavily_client(self.settings.tavily_api_key)
                )
                results = tavily.invoke(query)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return f"Web search failed: {exc}"

            if isinstance(results, dict):
                raw_results = results.get("results", [])
            elif isinstance(results, list):
                raw_results = results
            elif isinstance(results, str):
                return results
            else:
                raw_results = [str(results)]

            formatted = format_tavily_results(raw_results)
            return "\n\n".join(formatted)

        @function_tool
        def math_web_search(query: str) -> str:
            """Search curated math-oriented web sources for tutoring context."""
            return _run_tavily_search(query, math_only=True)

        @function_tool
        def web_search(query: str) -> str:
            """Search the open web for general research questions."""
            return _run_tavily_search(query, math_only=False)

        model_settings = ModelSettings(
            temperature=self.settings.agent_temperature,
            parallel_tool_calls=False,
        )

        router_agent = Agent(
            name="RouterAgent",
            handoff_description=(
                "Classifies the request into math tutoring or general web research."
            ),
            instructions=(
                "You are the top-level router. "
                "If the user is asking to solve, explain, derive, prove, simplify, or teach a "
                "mathematics problem, handoff to SolverAgent. "
                "If the user is asking for research about papers, methods, architectures, "
                "history, current information, or topics that are not a direct math tutoring "
                "problem, handoff to WebResearchAgent. "
                "Do not answer the question yourself."
            ),
            model=self.settings.triage_model,
            model_settings=model_settings,
        )

        solver_agent = Agent(
            name="SolverAgent",
            handoff_description=(
                "Primary math tutor that can solve directly or delegate research "
                "before producing the final answer."
            ),
            instructions=(
                "You are the primary mathematics tutor. Solve the user's problem directly "
                "when you already have enough information. "
                "If you need a theorem statement, worked example, retrieved reference, or "
                "curated math web context, handoff to MathResearchAgent. "
                "When MathResearchAgent hands back context, continue the solution yourself and "
                "produce the final answer with a concise step-by-step explanation. "
                "Do not answer non-mathematical requests."
            ),
            model=self.settings.solver_model,
            model_settings=model_settings,
        )

        math_research_agent = Agent(
            name="MathResearchAgent",
            handoff_description=(
                "Gathers tutoring context from the math knowledge base and curated math web."
            ),
            instructions=(
                "You are a math research specialist. Use the available tools to gather only "
                "the minimum context needed to help solve the user's math problem. "
                "Prefer the local knowledge base first, and use curated math web search only "
                "when the knowledge base is insufficient. "
                "After collecting relevant tutoring context, handoff back to SolverAgent with a short "
                "research brief. Do not produce the final user-facing answer yourself."
            ),
            model=self.settings.research_model,
            model_settings=model_settings,
            tools=[search_knowledge_base, math_web_search],
            handoffs=[solver_agent],
        )

        web_research_agent = Agent(
            name="WebResearchAgent",
            handoff_description=(
                "Handles general research questions using web search only."
            ),
            instructions=(
                "You handle general research questions that are not direct math tutoring tasks. "
                "Use web_search to gather evidence from the web, then answer the user clearly. "
                "Do not use or request the local math tutoring knowledge base. "
                "Summarize findings directly for the user and note uncertainty when sources are thin."
            ),
            model=self.settings.research_model,
            model_settings=model_settings,
            tools=[web_search],
        )

        router_agent.handoffs.extend([solver_agent, web_research_agent])
        solver_agent.handoffs.append(math_research_agent)

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
            router_agent=router_agent,
            solver_agent=solver_agent,
            math_research_agent=math_research_agent,
            web_research_agent=web_research_agent,
            run_config=run_config,
        )

    def _model_for_agent(self, agent_name: str) -> str:
        """Return the configured model name for a given runtime agent."""
        if agent_name == self._runtime.router_agent.name:
            return self.settings.triage_model
        if agent_name == self._runtime.solver_agent.name:
            return self.settings.solver_model
        return self.settings.research_model

    def _stringify_tool_output(self, output: Any) -> str:
        """Convert tool output to a compact, frontend-safe preview string."""
        if isinstance(output, str):
            text = output
        else:
            try:
                text = json.dumps(output, ensure_ascii=False, default=str)
            except TypeError:
                text = str(output)

        text = " ".join(text.split())
        return text[:1200]

    async def ask(self, question: str) -> BackendResponse:
        """Run the streamed agent workflow and return the final assembled response."""
        final_event: dict[str, Any] | None = None

        async for event in self.stream_ask(question):
            if event["type"] == "done":
                final_event = event

        if final_event is None:
            raise RuntimeError("Agent run ended without a final answer.")

        return BackendResponse(
            answer=final_event["answer"],
            reasoning=final_event["reasoning"],
            backend=self.backend_name,
            metadata=final_event["metadata"],
        )

    async def stream_ask(self, question: str):
        """Yield SSE-ready events from one streamed agent run."""
        from agents.items import HandoffOutputItem, ToolCallItem, ToolCallOutputItem
        from agents.stream_events import (
            AgentUpdatedStreamEvent,
            RawResponsesStreamEvent,
            RunItemStreamEvent,
        )

        validated_question = self._input_guard(question)
        reasoning_chunks: list[str] = []
        answer_chunks: list[str] = []
        research_used = False
        current_agent_name = self._runtime.router_agent.name

        run_result = self._runtime.runner.run_streamed(
            self._runtime.router_agent,
            validated_question,
            run_config=self._runtime.run_config,
        )

        yield {
            "type": "meta",
            "backend": self.backend_name,
            "model": self._model_for_agent(current_agent_name),
            "research_used": research_used,
            "last_agent": current_agent_name,
        }

        answer_agent_names = {
            self._runtime.solver_agent.name,
            self._runtime.web_research_agent.name,
        }
        async for event in run_result.stream_events():
            if isinstance(event, AgentUpdatedStreamEvent):
                current_agent_name = event.new_agent.name
                research_used = research_used or (
                    current_agent_name in {
                        self._runtime.math_research_agent.name,
                        self._runtime.web_research_agent.name,
                    }
                )
                yield {"type": "agent", "name": current_agent_name}
                continue

            if isinstance(event, RunItemStreamEvent):
                if event.name == "handoff_occured" and isinstance(event.item, HandoffOutputItem):
                    source_agent = getattr(event.item.source_agent, "name", current_agent_name)
                    target_agent = getattr(event.item.target_agent, "name", current_agent_name)
                    research_used = research_used or (
                        target_agent in {
                            self._runtime.math_research_agent.name,
                            self._runtime.web_research_agent.name,
                        }
                    )
                    yield {
                        "type": "handoff",
                        "from_agent": source_agent,
                        "to_agent": target_agent,
                    }
                    continue

                if event.name == "tool_called" and isinstance(event.item, ToolCallItem):
                    raw_item = event.item.raw_item
                    tool_name = getattr(raw_item, "name", None) or "tool"
                    if tool_name.startswith("transfer_to_"):
                        continue

                    research_used = True
                    yield {
                        "type": "tool_call",
                        "agent": getattr(event.item.agent, "name", current_agent_name),
                        "tool_name": tool_name,
                        "arguments": getattr(raw_item, "arguments", None) or "",
                        "call_id": getattr(raw_item, "call_id", None),
                    }
                    continue

                if event.name == "tool_output" and isinstance(event.item, ToolCallOutputItem):
                    raw_item = event.item.raw_item
                    tool_name = None
                    call_id = None
                    if isinstance(raw_item, dict):
                        tool_name = raw_item.get("name")
                        call_id = raw_item.get("call_id")
                    else:
                        tool_name = getattr(raw_item, "name", None)
                        call_id = getattr(raw_item, "call_id", None)

                    if (tool_name or "").startswith("transfer_to_"):
                        continue

                    yield {
                        "type": "tool_output",
                        "agent": getattr(event.item.agent, "name", current_agent_name),
                        "tool_name": tool_name or "tool",
                        "call_id": call_id,
                        "text": self._stringify_tool_output(event.item.output),
                    }
                    continue

            if not isinstance(event, RawResponsesStreamEvent):
                continue

            data = event.data
            data_type = getattr(data, "type", "")

            if current_agent_name not in answer_agent_names:
                continue

            if data_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                reasoning_text = getattr(data, "delta", None)
                if reasoning_text:
                    reasoning_chunks.append(reasoning_text)
                    yield {"type": "reason", "text": reasoning_text}
                continue

            if data_type in {"response.output_text.delta", "response.refusal.delta"}:
                answer_text = getattr(data, "delta", None)
                if answer_text:
                    answer_chunks.append(answer_text)
                    yield {"type": "answer", "text": answer_text}

        final_output = str(run_result.final_output).strip() if run_result.final_output else ""
        final_answer = self._output_guard(final_output or "".join(answer_chunks).strip())
        final_reasoning = "".join(reasoning_chunks).strip()
        last_agent_name = getattr(run_result.last_agent, "name", current_agent_name)
        yield {
            "type": "done",
            "answer": final_answer,
            "reasoning": final_reasoning,
            "metadata": {
                "backend": self.backend_name,
                "last_agent": last_agent_name,
                "model": self._model_for_agent(last_agent_name),
                "research_used": research_used,
            },
        }
