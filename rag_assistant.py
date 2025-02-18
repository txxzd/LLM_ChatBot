# rag_assistant.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Set your OpenAI API key here
openai.api_key = ""

def load_index(index_path="faiss_index.bin", meta_path="metadata.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def retrieve(query, index, metadata, model, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        results.append(metadata[i])
    return results

def generate_answer(query, context, model_engine="gpt-4o", max_tokens=1000):
    # Build a conversation with a system and user message
    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds to the questions or requests of the user."},
        {"role": "user", "content": f"Respond to the user based on the context provided:\n\nContext:\n{context}\n\nRequest: {query}"}
    ]
    response = openai.chat.completions.create(
        model=model_engine,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    answer = response.choices[0].message.content.strip()
    return answer

def answer_query(query, index, metadata, model, top_k=5):
    # Retrieve the most relevant content from the index
    retrieved_docs = retrieve(query, index, metadata, model, top_k)
    # Combine the retrieved texts into one context string
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    answer = generate_answer(query, context)
    return answer, retrieved_docs

if __name__ == "__main__":
    index, metadata = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = input("Enter your question: ")
    answer, docs = answer_query(query, index, metadata, model)
    print("Answer:")
    print(answer)
