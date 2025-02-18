# test_retrieval.py
from rag_assistant import load_index, retrieve
from sentence_transformers import SentenceTransformer

def main():
    # Load the FAISS index and metadata from disk
    index, metadata = load_index()
    
    # Load the embedding model used for generating embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Define a test query
    query = "What is a binary search tree?"
    
    # Retrieve the top 5 most relevant documents using the RAG retrieval function
    retrieved_docs = retrieve(query, index, metadata, model, top_k=5)
    
    # Print out the results
    print("Query:", query)
    print("Retrieved Documents:")
    for idx, doc in enumerate(retrieved_docs, start=1):
        print(f"\nDocument {idx}:")
        print("Filename:", doc["filename"])
        # Print the first 300 characters of the document text as a snippet
        print("Snippet:", doc["text"][:300], "\n---")

if __name__ == "__main__":
    main()
