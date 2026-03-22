import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const STREAM_ENDPOINT = `${API_BASE}/ask/stream`;

const assistantWelcome = {
  id: "welcome",
  role: "assistant",
  text: "Ask a math question and I will stream the answer, show the solver's reasoning trail, and surface the runtime traces.",
  reasoning: "",
  trace: {
    backend: "agentic",
    model: "deepseek-reasoner",
    researchUsed: false,
  },
  status: "done",
  isReasoningOpen: false,
};

function formatMathContent(text) {
  return text
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$")
    .replace(/\\\[/g, "$$")
    .replace(/\\\]/g, "$$")
    .replace(/^(\d+\.)/gm, "\n**Step $1**\n")
    .replace(/\*\*Final Answer:\*\*(.*?)(\n|$)/g, '\n<div class="final-answer">**Final Answer:** $1</div>\n')
    .replace(/\[(Q_\d+[^\]]*)\]/g, '<span class="equation-label">$1</span>')
    .replace(/\n\s*\n/g, "\n\n");
}

function renderMarkdown(text) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeRaw, rehypeKatex]}
      components={{
        p: ({ children }) => {
          const content = String(children);
          if (content.includes("$") || content.includes("\\(") || content.includes("\\[")) {
            return <p className="math-paragraph">{children}</p>;
          }
          return <p>{children}</p>;
        },
        code: ({ inline, className, children }) => {
          if (inline) {
            return <code className={`inline-math ${className || ""}`}>{children}</code>;
          }
          return <code className={className}>{children}</code>;
        },
        ol: ({ children }) => <ol className="step-list">{children}</ol>,
        li: ({ children }) => <li className="step-item">{children}</li>,
      }}
    >
      {formatMathContent(text)}
    </ReactMarkdown>
  );
}

function createAssistantMessage(id) {
  return {
    id,
    role: "assistant",
    text: "",
    reasoning: "",
    trace: {
      backend: "",
      model: "",
      researchUsed: null,
    },
    status: "streaming",
    isReasoningOpen: true,
  };
}

function parseSsePayload(buffer) {
  const events = [];
  const segments = buffer.split("\n\n");
  const rest = segments.pop() ?? "";

  for (const segment of segments) {
    const lines = segment
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .filter(Boolean);

    if (!lines.length) {
      continue;
    }

    try {
      events.push(JSON.parse(lines.join("\n")));
    } catch {
      // Ignore malformed partial chunks.
    }
  }

  return { events, rest };
}

function TracePill({ label, value, accent = "default" }) {
  if (value === undefined || value === null || value === "") {
    return null;
  }

  return (
    <span className={`trace-pill trace-pill-${accent}`}>
      <span className="trace-pill-label">{label}</span>
      <span className="trace-pill-value">{String(value)}</span>
    </span>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([assistantWelcome]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  const metrics = {
    total: messages.length,
    solved: messages.filter((message) => message.role === "assistant" && message.status === "done").length,
    streamed: messages.filter((message) => message.role === "assistant" && message.reasoning).length,
  };

  const updateAssistantMessage = (id, updater) => {
    setMessages((prev) =>
      prev.map((message) => (message.id === id ? updater(message) : message)),
    );
  };

  const askQuestion = async (event) => {
    event.preventDefault();
    if (!question.trim() || isLoading) {
      return;
    }

    const userQuestion = question.trim();
    const assistantId = `assistant-${Date.now()}`;

    setQuestion("");
    setIsLoading(true);
    setMessages((prev) => [
      ...prev,
      { id: `user-${assistantId}`, role: "user", text: userQuestion },
      createAssistantMessage(assistantId),
    ]);

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(STREAM_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userQuestion }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server error (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const { events, rest } = parseSsePayload(buffer);
        buffer = rest;

        for (const streamEvent of events) {
          if (streamEvent.type === "meta") {
            updateAssistantMessage(assistantId, (message) => ({
              ...message,
              trace: {
                ...message.trace,
                backend: streamEvent.backend || message.trace.backend,
                researchUsed: streamEvent.research_used,
              },
            }));
            continue;
          }

          if (streamEvent.type === "reason") {
            updateAssistantMessage(assistantId, (message) => ({
              ...message,
              reasoning: `${message.reasoning}${streamEvent.text}`,
              isReasoningOpen: true,
            }));
            continue;
          }

          if (streamEvent.type === "answer") {
            updateAssistantMessage(assistantId, (message) => ({
              ...message,
              text: `${message.text}${streamEvent.text}`,
            }));
            continue;
          }

          if (streamEvent.type === "done") {
            updateAssistantMessage(assistantId, (message) => ({
              ...message,
              text: streamEvent.answer || message.text,
              reasoning: streamEvent.reasoning || message.reasoning,
              status: "done",
              trace: {
                ...message.trace,
                backend: streamEvent.metadata?.backend || message.trace.backend,
                model: streamEvent.metadata?.model || message.trace.model,
                researchUsed: streamEvent.metadata?.research_used ?? message.trace.researchUsed,
                lastAgent: streamEvent.metadata?.last_agent,
              },
            }));
            continue;
          }

          if (streamEvent.type === "error") {
            updateAssistantMessage(assistantId, (message) => ({
              ...message,
              text: streamEvent.text,
              status: "error",
            }));
          }
        }
      }
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }

      updateAssistantMessage(assistantId, (message) => ({
        ...message,
        text: error.message?.startsWith("Server error")
          ? error.message
          : "Network error: unable to reach the backend stream.",
        status: "error",
      }));
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const clearChat = () => {
    abortControllerRef.current?.abort();
    setIsLoading(false);
    setMessages([assistantWelcome]);
  };

  const toggleReasoning = (id) => {
    updateAssistantMessage(id, (message) => ({
      ...message,
      isReasoningOpen: !message.isReasoningOpen,
    }));
  };

  return (
    <div className="app-shell">
      <div className="background-orb orb-a" />
      <div className="background-orb orb-b" />
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">∑</div>
          <div>
            <p className="eyebrow">Agentic Math Runtime</p>
            <h1 className="main-title">Trace the reasoning, not just the result.</h1>
          </div>
        </div>
        <button onClick={clearChat} className="clear-btn" disabled={isLoading}>
          Reset Session
        </button>
      </header>

      <main className="workspace">
        <section className="overview-panel">
          <div className="hero-card">
            <p className="hero-kicker">Live DeepSeek reasoner stream</p>
            <h2>Watch the solution, the traces, and the solver’s thinking arrive in real time.</h2>
            <p className="hero-copy">
              The backend now emits streamed reasoning and trace metadata. This interface surfaces both,
              while keeping the final answer readable on desktop and mobile.
            </p>
          </div>

          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">Messages</span>
              <strong className="stat-value">{metrics.total}</strong>
            </div>
            <div className="stat-card">
              <span className="stat-label">Solved</span>
              <strong className="stat-value">{metrics.solved}</strong>
            </div>
            <div className="stat-card">
              <span className="stat-label">Reasoning Streams</span>
              <strong className="stat-value">{metrics.streamed}</strong>
            </div>
          </div>
        </section>

        <section className="chat-shell">
          <div className="messages-container">
            {messages.map((message) => (
              <article
                key={message.id}
                className={`message-card ${message.role === "user" ? "message-user" : "message-assistant"}`}
              >
                <div className={`avatar ${message.role === "user" ? "avatar-user" : "avatar-assistant"}`}>
                  {message.role === "user" ? "You" : "AI"}
                </div>

                <div className={`message-body ${message.role === "user" ? "bubble-user" : "bubble-assistant"} ${message.status === "error" ? "bubble-error" : ""}`}>
                  <div className="message-topline">
                    <span className="role-label">{message.role === "user" ? "User Prompt" : "Agent Response"}</span>
                    {message.role === "assistant" && (
                      <span className={`status-chip status-${message.status || "done"}`}>
                        {message.status === "streaming" ? "Streaming" : message.status === "error" ? "Error" : "Complete"}
                      </span>
                    )}
                  </div>

                  {message.role === "assistant" && (
                    <div className="trace-row">
                      <TracePill label="Runtime" value={message.trace?.backend} accent="neutral" />
                      <TracePill label="Model" value={message.trace?.model} accent="warm" />
                      <TracePill
                        label="Research"
                        value={
                          message.trace?.researchUsed === null || message.trace?.researchUsed === undefined
                            ? ""
                            : message.trace.researchUsed
                              ? "Used"
                              : "Skipped"
                        }
                        accent={message.trace?.researchUsed ? "active" : "default"}
                      />
                      <TracePill label="Last Agent" value={message.trace?.lastAgent} accent="neutral" />
                    </div>
                  )}

                  <div className="message-content">
                    {message.text ? renderMarkdown(message.text) : message.role === "assistant" ? (
                      <div className="loading-panel">
                        <div className="pulse-bar" />
                        <div className="pulse-bar short" />
                        <p>Waiting for answer tokens...</p>
                      </div>
                    ) : null}
                  </div>

                  {message.role === "assistant" && (message.reasoning || message.status === "streaming") && (
                    <div className="reasoning-panel">
                      <button
                        type="button"
                        className="reasoning-toggle"
                        onClick={() => toggleReasoning(message.id)}
                      >
                        <span>Reasoning Trace</span>
                        <span>{message.isReasoningOpen ? "Hide" : "Show"}</span>
                      </button>
                      {message.isReasoningOpen && (
                        <div className="reasoning-content">
                          {message.reasoning ? (
                            <pre>{message.reasoning}</pre>
                          ) : (
                            <p className="reasoning-placeholder">Reasoning tokens are still arriving.</p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </article>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={askQuestion} className="composer">
            <div className="composer-copy">
              <p className="composer-label">Prompt</p>
              <p className="composer-hint">
                Try a direct solve, a proof sketch, or a concept question that triggers research.
              </p>
            </div>
            <div className="composer-controls">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Example: Derive the quadratic formula from ax² + bx + c = 0."
                className="question-input"
                rows={3}
                disabled={isLoading}
              />
              <div className="composer-actions">
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => setQuestion("")}
                  disabled={isLoading || !question}
                >
                  Clear
                </button>
                <button
                  type="submit"
                  className={`submit-btn ${isLoading || !question.trim() ? "disabled" : ""}`}
                  disabled={isLoading || !question.trim()}
                >
                  {isLoading ? "Streaming…" : "Send Prompt"}
                </button>
              </div>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;
