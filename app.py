import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

from agentturing.guardrails.setup import make_output_guard, make_input_guard
from agentturing.pipelines.main_pipeline import build_graph

print("Bootstrapping pipeline (this happens once)...")
GRAPH = build_graph()
INPUT_GUARD = make_input_guard()
OUTPUT_GUARD = make_output_guard()
print("Pipeline ready.")

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Math Tutor API", version="1.0")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_math(request: QueryRequest):
    question = request.question.strip()
    print(question)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1) Input guard
    try:
        validated_question = INPUT_GUARD(question)
    except Exception as e:
        return {
            "answer": "This assistant only handles mathematics questions. Please provide a math-related query.",
            "error": f"Input guard triggered: {str(e)}"
        }
    # 2) Run pipeline
    try:
        state = {"question": validated_question}
        result = GRAPH.invoke(state)

        raw_answer = result["answer"][0]["generated_text"]
        answer = raw_answer.partition("Answer101:")[2].strip() or raw_answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # 3) Output guard
    try:
        safe_answer = OUTPUT_GUARD(answer)
    except Exception:
        safe_answer = "The generated answer did not meet safety requirements. Please rephrase the question."

    return {"question": question, "answer": safe_answer}
