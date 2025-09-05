import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "./App.css"; // Import our CSS file

function App() {
  const formatMathContent = (text) => {
  // Better LaTeX delimiter handling
  return text
    .replace(/\\\(/g, '$')
    .replace(/\\\)/g, '$')
    .replace(/\\\[/g, '$$')
    .replace(/\\\]/g, '$$')
    // Handle step numbering
    .replace(/^(\d+\.)/gm, '\n**Step $1**\n')
    // Handle final answer formatting
    .replace(/\*\*Final Answer:\*\*(.*?)(\n|$)/g, '\n<div class="final-answer">**Final Answer:** $1</div>\n')
    // Handle equation labels
    .replace(/\[(Q_\d+[^\]]*)\]/g, '<span class="equation-label">$1</span>')
    // Clean up extra whitespace
    .replace(/\n\s*\n/g, '\n\n');
};
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "👋 Hello! I'm your mathematics tutor. Ask me any math question and I'll provide step-by-step solutions!",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const askQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    const userQuestion = question.trim();
    setQuestion("");
    setIsLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: userQuestion }]);

    const loadingMessage = { role: "assistant", text: "🤔 Thinking...", isLoading: true };
    setMessages((prev) => [...prev, loadingMessage]);

    try {
      const res = await axios.post("http://localhost:8000/ask", {
        question: userQuestion,
      });

      setMessages((prev) => {
        const filtered = prev.filter((m) => !m.isLoading);
        const answer = res.data.answer || "I apologize, but I couldn't generate a response.";
        return [...filtered, { role: "assistant", text: answer }];
      });

      if (res.data.error) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            text: `⚠️ ${res.data.error}`,
            isError: true,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => {
        const filtered = prev.filter((m) => !m.isLoading);
        let errorMessage = "❌ I'm having trouble connecting to the server. Please try again.";
        
        if (error.response) {
          errorMessage = `❌ Server error (${error.response.status}): ${
            error.response.data?.detail || "Please try again later."
          }`;
        } else if (error.request) {
          errorMessage = "❌ Network error: Please check your internet connection.";
        }

        return [...filtered, { role: "assistant", text: errorMessage, isError: true }];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: "assistant",
        text: "👋 Hello! I'm your mathematics tutor. Ask me any math question and I'll provide step-by-step solutions!",
      },
    ]);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo">🧮</div>
            <div className="title-section">
              <h1 className="main-title">AgentTuring Math Bot</h1>
              <p className="subtitle">Your AI Mathematics Tutor</p>
            </div>
          </div>
          <button
            onClick={clearChat}
            className="clear-btn"
            disabled={isLoading}
          >
            Clear Chat
          </button>
        </div>
      </header>

      {/* Main Chat Container */}
      <main className="main-container">
        <div className="chat-container">
          
          {/* Messages */}
          <div className="messages-container">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`message ${m.role === "user" ? "user" : "assistant"}`}
              >
                {/* Avatar */}
                <div className={`avatar ${m.role === "user" ? "user" : "assistant"}`}>
                  {m.role === "user" ? "👤" : "🤖"}
                </div>

                {/* Message Bubble */}
                <div
                  className={`bubble ${
                    m.role === "user"
                      ? "user"
                      : m.isError
                      ? "error"
                      : m.isLoading
                      ? "loading"
                      : "assistant"
                  }`}
                >
                  {m.isLoading ? (
                    <div className="loading-content">
                      <div className="load-dots">
                        <div className="load-dot"></div>
                        <div className="load-dot"></div>
                        <div className="load-dot"></div>
                      </div>
                      <span>Working on your problem...</span>
                    </div>
                  ) : (
                    <div className="message-content">
  <ReactMarkdown
    remarkPlugins={[remarkMath]}
    rehypePlugins={[rehypeKatex]}
    components={{
      p: ({ children }) => {
        const content = String(children);
        if (content.includes('$') || content.includes('\\(') || content.includes('\\[')) {
          return <p className="math-paragraph">{children}</p>;
        }
        return <p>{children}</p>;
      },
      code: ({ inline, className, children }) => {
        if (inline) {
          return <code className={`inline-math ${className || ''}`}>{children}</code>;
        }
        return <code className={className}>{children}</code>;
      },
      // Better list styling for steps
      ol: ({ children }) => <ol className="step-list">{children}</ol>,
      li: ({ children }) => <li className="step-item">{children}</li>,
    }}
  >
    {formatMathContent(m.text)}
  </ReactMarkdown>
</div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form onSubmit={askQuestion} className="input-form">
            <div className="input-container">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask me any math question... (e.g., Solve x² + 5x - 6 = 0)"
                className="question-input"
                disabled={isLoading}
              />
              {question && (
                <button
                  type="button"
                  onClick={() => setQuestion("")}
                  className="clear-input-btn"
                  disabled={isLoading}
                >
                  ✕
                </button>
              )}
            </div>
            <button
              type="submit"
              disabled={isLoading || !question.trim()}
              className={`submit-btn ${
                isLoading || !question.trim() ? "disabled" : ""
              }`}
            >
              {isLoading ? (
                <>
                  <div className="loading-spinner"></div>
                  <span className="btn-text">Solving</span>
                </>
              ) : (
                <>
                  <span className="btn-text">Send</span>
                  <span className="btn-icon">📤</span>
                </>
              )}
            </button>
          </form>
          
          <p className="helper-text">
            I can solve equations, explain concepts, and provide step-by-step solutions for mathematics problems.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
