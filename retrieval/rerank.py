class HeuristicReranker:
    def rerank(self, query: str, results: list, top_k: int) -> list:
        max_cos = max((r["cosine"] for r in results), default=1.0) or 1.0
        for r in results:
            r["rerank_score"] = r["rrf"] * (0.5 + 0.5 * (r["cosine"] / max_cos))
        return sorted(results, key=lambda x: -x["rerank_score"])[:top_k]

def get_reranker():
    try:
        from sentence_transformers import CrossEncoder
        class CrossReranker:
            def __init__(self):
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            def rerank(self, query: str, results: list, top_k: int) -> list:
                if not results: return []
                pairs = [(query, r["text"]) for r in results]
                scores = self.model.predict(pairs, show_progress_bar=False)
                for r, s in zip(results, scores): r["rerank_score"] = float(s)
                return sorted(results, key=lambda x: -x["rerank_score"])[:top_k]
        print("[INFO] Cross-Encoder active.")
        return CrossReranker()
    except Exception:
        print("[INFO] Heuristic Reranker active.")
        return HeuristicReranker()