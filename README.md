# LLM_ChatBot
 
Personalized NLP Learning Assistant using RAG
This project implements a personalized learning assistant for Natural Language Processing (NLP) using a Retrieval-Augmented Generation (RAG) framework. The assistant ingests course materials (text and PDFs), vectorizes them using sentence embeddings, and builds a searchable FAISS index. When a user asks a question, the system retrieves the most relevant content and uses a large language model (LLM) to generate a context-aware answer. An interactive Streamlit-based chat interface provides a ChatGPT-like experience for users.

## Features
### Content Ingestion:
Automatically reads course materials from a specified folder. Supports both plain text (.txt) and PDF (.pdf) files using PyMuPDF.

### Text Chunking and Vectorization:
Splits large documents into manageable, overlapping chunks. Converts text into embeddings using Sentence Transformers.

### FAISS Indexing:
Builds a FAISS index for fast retrieval of relevant content based on semantic similarity.

### Retrieval-Augmented Generation (RAG):
Uses the retrieved context to prompt an LLM (via the OpenAI API) to generate detailed and context-aware answers.

### Interactive Chat Interface:
Implements a chat UI using Streamlit that mimics ChatGPT, preserving conversation history and displaying relevant source documents.

### Evaluation Metrics:
Supports evaluation of generated responses using BLEU and BERTScore metrics for quantitative analysis.

## Project Structure
### vectorize_materials.py
Script to load, chunk, and vectorize course materials. Builds and saves the FAISS index and corresponding metadata.

### rag_assistant.py
Contains the RAG pipeline: loading the FAISS index, retrieving relevant content, and generating answers using the OpenAI API with the new namespaced interface.

### app.py
A Streamlit application providing a chat interface for interacting with the learning assistant. Uses a ChatGPT-like layout to display conversation history and retrieved document metadata.

### evaluation.py
A script to evaluate the quality of the generated responses against reference answers using BLEU and BERTScore metrics.

### evaluation_dataset.json (example file)
JSON file containing test cases (queries and reference answers) for evaluation purposes.

