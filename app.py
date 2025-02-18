import streamlit as st
from rag_assistant import load_index, answer_query
from sentence_transformers import SentenceTransformer

st.title("Programming Course Assistant")

# Load resources only once at startup
@st.cache_resource(show_spinner=False)
def load_resources():
    index, metadata = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model

index, metadata, model = load_resources()

# Initialize conversation history in session state as a list of messages.
# Each message is a dictionary with keys: "role" and "content".
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display conversation history using Streamlit's chat_message component
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optionally display retrieved documents for assistant messages.
        if message["role"] == "assistant" and "docs" in message:
            st.markdown("**Retrieved Documents:**")
            for doc in message["docs"]:
                st.markdown(f"- {doc['filename']}")

# Chat input for user prompt
prompt = st.chat_input("Enter your question...")

if prompt:
    # Append user's message to conversation state
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve answer and documents using the RAG pipeline
    answer, docs = answer_query(prompt, index, metadata, model)
    
    # Append assistant's response (with retrieved docs) to conversation state
    st.session_state.conversation.append({"role": "assistant", "content": answer, "docs": docs})
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown("**Retrieved Documents:**")
        for doc in docs:
            st.markdown(f"- {doc['filename']}")
