import streamlit as st
from rag_assistant import load_index, answer_query, generate_answer
from sentence_transformers import SentenceTransformer

st.title("Personalized NLP Learning Assistant")

# Load resources only once at startup
@st.cache_resource(show_spinner=False)
def load_resources():
    index, metadata = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model

index, metadata, model = load_resources()

# Initialize conversation history in session state if not present.
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display conversation history using Streamlit's chat_message component
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optionally show document metadata for assistant messages.
        if message["role"] == "assistant" and "docs" in message:
            st.markdown("**Retrieved Documents:**")
            for doc in message["docs"]:
                st.markdown(f"- {doc['filename']}")

# Chat input for user prompt.
current_query = st.chat_input("Enter your question...")

if current_query:
    # Append user's message to conversation state.
    st.session_state.conversation.append({"role": "user", "content": current_query})
    with st.chat_message("user"):
        st.markdown(current_query)
    
    # Retrieve documents using the current query (retrieval remains unchanged)
    answer, docs = answer_query(current_query, index, metadata, model)
    
    # Build conversation history from the last five messages.
    conversation_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation[-5:]
    )
    
    # Construct a new prompt that incorporates the conversation history.
    full_prompt = f"Here is our conversation so far:\n{conversation_history}\nUser: {current_query}\nAssistant:"
    
    # Combine the text from the retrieved documents as context.
    retrieved_context = "\n\n".join([doc["text"] for doc in docs])
    
    # Generate a new answer using the full prompt (including conversation history) and the retrieved context.
    new_answer = generate_answer(full_prompt, retrieved_context)
    
    # Append the assistant's response (with document metadata) to the conversation.
    st.session_state.conversation.append({"role": "assistant", "content": new_answer, "docs": docs})
    with st.chat_message("assistant"):
        st.markdown(new_answer)
        st.markdown("**Retrieved Documents:**")
        for doc in docs:
            st.markdown(f"- {doc['filename']}")
