class ConversationalAgent:
    """
    Thin wrapper around EngineeringRAG.

    The naive 'and' / 'vs' split that previously decomposed compound
    questions into two independent rag.answer() calls has been REMOVED.
    The extractive sentence selector (in the RAG engine) handles multi-part
    questions in a single pass by selecting 2-3 sentences from the retrieved
    chunks — no route split, no risk of one half picking the wrong SQL table.

    Pronoun resolution is guarded so it works even if the optional
    graph_rag module isn't present.
    """

    def __init__(self, rag):
        self.rag = rag
        self.last_doc_id = None

    def ask(self, query: str, user) -> dict:
        q_lower = query.lower()

        # Follow-up detection: short query with pronoun reference
        is_followup = (
            len(query.split()) < 8
            and any(marker in q_lower for marker in
                    [" it", " it?", " it.", " that", " the fix", " the issue"])
        )

        resolved_q = query
        if is_followup and self.last_doc_id and hasattr(self.rag, "graph_rag"):
            try:
                resolved_q = self.rag.graph_rag.resolve_pronoun(self.last_doc_id, query)
                print(f"[GRAPH RAG] Resolved '{query}' -> '{resolved_q}'")
            except Exception as e:
                print(f"[GRAPH RAG] Pronoun resolution failed: {e}")

        # Delegate to RAG (compound questions handled by extractive selector)
        res = self.rag.answer(resolved_q, user)

        if res.get("sources"):
            self.last_doc_id = res["sources"][0]["doc_id"]

        res["resolved_query"] = resolved_q
        return res

    def reset(self):
        """Clear conversation memory (called by the Streamlit 'clear' button)."""
        self.last_doc_id = None