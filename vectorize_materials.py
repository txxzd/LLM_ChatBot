# vectorize_materials.py
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
        elif filename.endswith(".pdf"):
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            documents.append({"filename": filename, "text": text})
    return documents
def chunk_text(text, chunk_size=500, overlap=100):
    # A simple overlapping chunking mechanism
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def create_index(documents, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = []
    metadata = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        texts.extend(chunks)
        metadata.extend([{"filename": doc["filename"], "text": chunk} for chunk in chunks])
    # Encode texts into embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    # Create a FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, metadata, model

def save_index(index, metadata, file_path="faiss_index.bin", meta_path="metadata.json"):
    faiss.write_index(index, file_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    # Ensure your course materials are in the "course_materials" folder.
    documents = load_documents("course_materials")
    index, metadata, model = create_index(documents)
    save_index(index, metadata)
    print("Index and metadata saved.")