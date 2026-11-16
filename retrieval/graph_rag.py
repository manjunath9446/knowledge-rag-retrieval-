import networkx as nx
import re

class KnowledgeGraphRAG:
    def __init__(self, docs, embedder):
        self.graph = nx.DiGraph()
        self.embedder = embedder
        self.canonical_nodes = {}
        self._build_graph(docs)

    def _resolve_entity(self, entity_text: str) -> str:
        entity_text = entity_text.lower().strip()
        if not entity_text or len(entity_text) < 3: return entity_text
        if entity_text in self.canonical_nodes: return self.canonical_nodes[entity_text]
        canonical_id = entity_text.replace(" ", "_")
        self.canonical_nodes[entity_text] = canonical_id
        return canonical_id

    def _build_graph(self, docs):
        for d in docs:
            doc_id = d["doc_id"]
            title = d["title"]
            self.graph.add_node(doc_id, type="Document", title=title)
            
            if re.search(r"decision|action|resolution|fix", title, re.IGNORECASE):
                role = "resolution"
            elif re.search(r"investigation|rca|report|failure", title, re.IGNORECASE):
                role = "root_cause"
            else:
                role = "general"
                
            self.graph.add_edge(doc_id, role, relation="HAS_ROLE")
            entity = self._resolve_entity(title)
            self.graph.add_edge(doc_id, entity, relation="ABOUT")
            
            for issue in ["flashback", "erosion", "stress", "leak"]:
                if issue in d["content"].lower():
                    self.graph.add_edge(self._resolve_entity(issue), doc_id, relation="DOCUMENTED_IN")

    def resolve_pronoun(self, last_doc_id: str, query: str) -> str:
        if not last_doc_id or last_doc_id not in self.graph:
            return query
            
        predecessors = list(self.graph.predecessors(last_doc_id))
        if predecessors:
            entity = predecessors[0].replace("_", " ")
            if len(entity) > 30: entity = entity[:30] + "..."
            
            if "the fix" in query.lower():
                # Explicitly route to the Resolution role
                return f"resolution for {entity}"
            elif " it" in query.lower():
                return query.lower().replace(" it", f" the {entity}")
        return query