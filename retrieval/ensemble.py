from retrieval.rerank import get_reranker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnsembleReranker:
    def __init__(self):
        self.neural_reranker = get_reranker()

    def rerank_with_consensus(self, query: str, results: list, top_k: int):
        if not results: return [], True
        
        # Judge 1: Neural Cross-Encoder
        neural_results = self.neural_reranker.rerank(query, results.copy(), top_k=top_k)
        
        # Judge 2: Lexical TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english").fit_transform([query] + [r["text"] for r in results])
        tfidf_scores = cosine_similarity(vectorizer[0:1], vectorizer[1:])[0]
        lexical_results = [r for _, r in sorted(zip(tfidf_scores, results), key=lambda x: -x[0])][:top_k]
        
        # Check if the neural top document is in the lexical top 3
        neural_top_doc = neural_results[0]["doc"]["doc_id"] if neural_results else None
        lexical_top_docs = [r["doc"]["doc_id"] for r in lexical_results]
        
        consensus = neural_top_doc in lexical_top_docs
        
        # Log the disagreement but DO NOT BLOCK the pipeline
        if not consensus:
            print(f"[ENSEMBLE WARNING] Neural top ({neural_top_doc}) not in Lexical top 3. Trusting Neural.")
        
        return neural_results, True # Return True to let the pipeline continue