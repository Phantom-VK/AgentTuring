"""FastAPI application exposing agentic math endpoints."""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentturing.config import get_settings
from agentturing.services import AgenticBackendUnavailable, get_chat_backend

# Some personal laptop  / environment related settings, can remove
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SETTINGS = get_settings()

# FastAPI setup
app = FastAPI(title="Math Tutor API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):  # pylint: disable=too-few-public-methods
    """Incoming request payload for math questions."""

    question: str

@app.post("/ask")
async def ask_math(request: QueryRequest):
    """Return a complete answer and captured reasoning for a math prompt."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        backend = get_chat_backend()
        response = await backend.ask(question)
    except ValueError as exc:
        return {
            "answer": (
                "This assistant only handles mathematics questions. "
                "Please provide a math-related query."
            ),
            "error": f"Input guard triggered: {str(exc)}",
        }
    except AgenticBackendUnavailable as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}") from exc

    return {
        "question": question,
        "answer": response.answer,
        "reasoning": response.reasoning,
        "backend": response.backend,
        "metadata": response.metadata,
    }


@app.post("/ask/stream")
async def ask_math_stream(request: QueryRequest):
    """Stream reasoning and answer chunks as server-sent events."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        backend = get_chat_backend()
    except AgenticBackendUnavailable as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def event_stream():
        try:
            async for event in backend.stream_ask(question):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_text = f"Pipeline error: {str(exc)}"
            yield f"data: {json.dumps({'type': 'error', 'text': error_text})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
