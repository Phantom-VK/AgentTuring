import streamlit as st

from agentturing.pipelines.main_pipeline import build_graph

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Math Tutor Bot", page_icon="📐", layout="wide")

st.title("📐 Math Tutor Bot")
st.caption("An Agentic-RAG powered professor that solves math problems step by step.")

# Chat display
for role, message in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)

# Input box
if prompt := st.chat_input("Ask me a math question..."):
    # Append user message to chat history
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").markdown(prompt)

    # Run pipeline
    try:
        result = st.session_state.graph.invoke({"question": prompt})
        answer = result['answer'][0]['generated_text'].partition("Answer101:")[2]
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    st.session_state.history.append(("assistant", answer))
    st.chat_message("assistant").markdown(answer)
