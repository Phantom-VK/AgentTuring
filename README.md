
# AgentTuring Math Tutor

An AI-powered mathematics tutor built around the OpenAI Agents SDK, a DeepSeek OpenAI-compatible backend, local Qdrant retrieval, and Tavily-backed web search.

---

## 🚀 Features

- **Agentic math tutoring** with explicit tools and handoffs.
- **Knowledge Base retrieval** with vector search on MetaMathQA and Math-Step-DPO-10K datasets.
- **Dynamic Tavily web search** for out-of-knowledge queries.
- **Input/Output guards** to filter toxic content and PII, enforcing math-only queries.
- **React frontend** with LaTeX and Markdown rendering for rich math display.
- **DeepSeek via OpenAI-compatible `base_url`** with agent orchestration handled in the backend.

---

## 📂 Project Structure


```
agentturing/           \# Backend codebase
├─ config.py         \# Environment-driven backend settings
├─ constants/        \# Shared constants for embeddings, Qdrant, Tavily domains
├─ database/         \# KB construction and vector store management
├─ guardrails/       \# Input/output safety checks
├─ model/            \# Embedding model loader
├─ services/         \# Agent runtime, tools, and request handling
├─ utils/            \# Helper and search-output formatting

agentturing-frontend/  \# React frontend with chat UI and rendering
app.py                 \# FastAPI backend serving API endpoints
logs/                  \# Runtime and error logs
.env                   \# API Credentials, not pushed to Github
uv.lock                \# uv lockfile for Python dependencies
```

---

## 🔥 Setup & Installation

1. Clone the repo:
```

git clone https://github.com/phantom-vk/agentturing.git
cd agentturing

```

2. Sync dependencies with `uv`:
```

uv sync

```

3. Configure environment variables in `.env`:
```

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
OPENAI_DEFAULT_MODEL=deepseek-chat
AGENT_TRIAGE_MODEL=deepseek-chat
AGENT_SOLVER_MODEL=deepseek-reasoner
AGENT_RESEARCH_MODEL=deepseek-chat
TAVILY_API_KEY=your_tavily_api_key

```

4. Ingest datasets into the local Qdrant knowledge base:
```

uv run python -m agentturing.database.setup_knowledgebase

```

5. Run the FastAPI backend:
```

uv run uvicorn app:app --reload

```

6. Start React frontend:
```

cd agentturing-frontend
npm install
npm run dev

```

Notes:
- The backend now supports an agentic runtime using the OpenAI Agents SDK with a DeepSeek OpenAI-compatible `base_url` and API key.
- Tavily is optional, but required if you want the web-search tool available in the agentic backend.
- Qdrant is local in this repo, so no Qdrant API key is required unless you later switch to a hosted Qdrant deployment.

---

## ⚙️ How It Works

1. User types a math question in the frontend and sends it to the FastAPI backend.  
2. The backend validates scope and sanitizes obvious PII.  
3. A triage agent decides whether the question can go directly to solving or needs tool-assisted research.  
4. The research agent can use local Qdrant retrieval and Tavily web search.  
5. The solver agent produces the final step-by-step answer.  
6. Output safety checks run before the answer is returned to the UI.

---

## 🧪 Sample Questions to Try

- Explain the concept of p-adic numbers  
- Find the critical points of \( f(x) = x^3 - 3x^2 + 2 \)  
- What is the integral of \( e^{-x}x^3 \) from 0 to 1?  
- How many times in July do both a bookstore (sale every 5 days) and shoe store (sale every 6 days) have sales on the same date?  
- What is love in mathematics?

---

## 🌱 Future Work

- Implement **Human-in-the-Loop feedback** with DSPy integration for continuous learning from user corrections.  
- Expand KB with more datasets and domains.  
- Improve reasoning by fine-tuning with domain-specific datasets.

---


## ✉️ Contact

Created by Vikramaditya – [vikramadityakhupse@gmail.com]  
Feedback welcome!

---

⭐ If you find this project useful, please give it a star!

