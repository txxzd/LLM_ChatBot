import json
from rag_assistant import load_index, answer_query
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

def evaluate_response(candidate, reference):
    # Compute BLEU score with smoothing (important for short texts)
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    
    # Compute BERTScore (F1 score is typically used as the summary metric)
    _, _, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
    return bleu, F1.item()

def main():
    # Load the FAISS index, metadata, and embedding model from your project
    index, metadata = load_index()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load your evaluation dataset (a JSON file with test queries and reference answers)
    with open("evaluation.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    results = []
    
    # Iterate over each test case
    for test in dataset:
        query = test["query"]
        reference = test["reference"]
        # Use your RAG pipeline to generate an answer for the query
        answer, docs = answer_query(query, index, metadata, model)
        
        # Evaluate the generated answer
        bleu, bert_f1 = evaluate_response(answer, reference)
        results.append({
            "query": query,
            "generated_answer": answer,
            "reference": reference,
            "bleu": bleu,
            "bert_f1": bert_f1
        })
        
        # Print results for this test case
        print("Query:", query)
        print("Generated Answer:", answer)
        print("Reference Answer:", reference)
        print("BLEU Score:", bleu)
        print("BERTScore F1:", bert_f1)
        print("-" * 40)
    
    # Optionally, save the evaluation results to a file
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
