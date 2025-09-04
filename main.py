import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
from agentturing.guardrails.setup import make_output_guard, make_input_guard
from agentturing.pipelines.main_pipeline import build_graph
from agentturing.utils.sanitize_output import extract_steps


print("Bootstrapping pipeline (this happens once)...")
GRAPH = build_graph()
INPUT_GUARD = make_input_guard()
OUTPUT_GUARD = make_output_guard()
print("Pipeline ready. Enter questions (type 'exit' to quit).")


def run_query(question: str):
    # 1) Input guard (math-only + safety). Must return str.
    # try:
    #     validated_question = INPUT_GUARD(question)
    # except Exception as e:
    #     return {
    #         "answer": "This assistant only handles mathematics questions. Please provide a math-related query.",
    #         "error": f"Input guard triggered: {str(e)}"
    #     }

    # 2) Run graph
    state = {"question": question}
    result = GRAPH.invoke(state)

    # 3) Output guard (sanitize final text). Must return str.

    # try:
    #     safe_answer = OUTPUT_GUARD(raw_answer)
    # except Exception:
    #     safe_answer = "The generated answer did not meet safety requirements. Please rephrase the question."

    return result['answer'][0]['generated_text'].partition("Answer step-by-step:")[2]


if __name__ == "__main__":
    # Simple REPL loop; no rebuilds between queries
    try:
        while True:
            query = input("\nQ > ").strip()
            if query.lower() in {"exit", "quit", ":q"}:
                print("Goodbye.")
                break
            out = run_query(query)
            print(f"A > {out}")
            if "error" in out:
                print(f"(guard) {out['error']}")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
