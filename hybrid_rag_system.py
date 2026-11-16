import json
import sqlite3
import pandas as pd
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import textwrap

# ==========================================
# 1. SETUP (Groq Configuration)
# ==========================================
XAI_API_KEY = "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" # Replace with your actual key

client = OpenAI(
    api_key=XAI_API_KEY, 
    base_url="https://api.groq.com/openai/v1" 
)

SCRIPT_DIR = Path(__file__).parent
CHUNKS_FILE = SCRIPT_DIR / "data" / "parsed_chunks.json"
DB_FILE = SCRIPT_DIR / "parser" / "hycosys_tables.db"

class SmartHybridRAG:
    def __init__(self):
        print("="*70)
        print(" 🚀 INITIALIZING SMART HYBRID RAG (GROQ + NEURAL + SYMBOLIC)")
        print("="*70)
        
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(f"Missing {CHUNKS_FILE}")
            
        with open(CHUNKS_FILE, "r") as f:
            self.chunks = json.load(f)
            
        print(f" ✅ Loaded {len(self.chunks)} structure-aware text chunks.")
        print(" ⏳ Loading BGE Embedding Model...")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.chunk_embeddings = self.embedder.encode([c["text"] for c in self.chunks], show_progress_bar=False)
        
        if not DB_FILE.exists():
            raise FileNotFoundError(f"Missing {DB_FILE}")
            
        self.db_conn = sqlite3.connect(DB_FILE)
        self.table_schemas = self._get_schemas()
        print(f" ✅ Loaded {len(self.table_schemas)} SQL table schemas: {list(self.table_schemas.keys())}\n")

    def _get_schemas(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        return {row[0]: row[1] for row in cursor.fetchall()}

    # ==========================================
    # 2. THE INTELLIGENT ROUTER
    # ==========================================
    def route_and_answer(self, query: str):
        print(f"\n{'='*70}")
        print(f" ❓ USER QUERY: {query}")
        print(f"{'='*70}")
        
        if "YOUR_GROQ_API_KEY" in client.api_key:
            print(" ⚠️  [WARNING] No Groq API key found. Defaulting to Vector Search.")
            return self.query_vector(query)

        schemas_text = "\n".join(self.table_schemas.values())
        
        prompt = f"""
        You are an aerospace data router. 
        Here are the SQL table schemas available in the database:
        {schemas_text}
        
        EXAMPLES:
        User: "What is A1?" -> {{"route": "SQL", "sql_query": "SELECT * FROM table_1 WHERE \\"#No\\" = 'A1'"}}
        User: "Why did the shaft fail?" -> {{"route": "VECTOR"}}
        
        User Query: "{query}"
        
        INSTRUCTIONS:
        1. If the query asks for a specific number, value, table data, or a location code (like A1, A2, A3), output a JSON object with:
           - "route": "SQL"
           - "sql_query": "SELECT ..." (Write valid SQLite syntax based on the schema)
        2. If the query asks for an explanation, cause, or conceptual information, output:
           - "route": "VECTOR"
        
        Output ONLY a valid JSON object. Do not include markdown formatting or extra text.
        """
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            raw_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            
            if not json_match:
                raise ValueError("No JSON found in response")
                
            decision = json.loads(json_match.group(0))
            route = decision.get("route")
            print(f" 🧠 GROQ ROUTER DECISION: {route}")
            
            if route == "SQL" and "sql_query" in decision:
                return self.execute_sql(decision["sql_query"], original_query=query)
            else:
                return self.query_vector(query)
                
        except Exception as e:
            print(f" ⚠️  [ROUTER ERROR] {e}. Falling back to Vector Search.")
            return self.query_vector(query)

    # ==========================================
    # 3. SYMBOLIC LAYER (SQL)
    # ==========================================
    def execute_sql(self, sql_query: str, original_query: str):
        print(f" 🔍 SQL ROUTE SELECTED")
        print(f"    Generated Query: {sql_query}")
        
        try:
            df = pd.read_sql_query(sql_query, self.db_conn)
            
            if df.empty:
                print(" ⚠️  [SQL MISS] Empty result. Falling back to NEURAL (Vector) Layer...")
                return self.query_vector(original_query)
                
            print("\n ✅ [SQL RESULT]")
            print("-" * 70)
            print(df.to_string(index=False))
            print("-" * 70)
            return df
            
        except Exception as e:
            print(f" ⚠️  [SQL ERROR] {e}. Falling back to NEURAL (Vector) Layer...")
            return self.query_vector(original_query)

    # ==========================================
    # 4. NEURAL LAYER (Vector)
    # ==========================================
    def query_vector(self, query: str):
        print(" 🔍 VECTOR SEARCH ROUTE SELECTED")
        q_vec = self.embedder.encode([f"Represent this sentence for searching relevant passages: {query}"])[0]
        scores = cosine_similarity([q_vec], self.chunk_embeddings)[0]
        top_idx = np.argmax(scores)
        
        best_chunk = self.chunks[top_idx]
        score = scores[top_idx]
        
        # TUNED THRESHOLD TO 0.70 (Allows CT Scan at 0.72, rejects Turbine Blade at 0.66)
        if score < 0.70:
            print(f"\n ❌ [VECTOR RESULT] Confidence too low ({score:.2f} < 0.70). Refusing to answer.")
            return "I don't have sufficiently grounded evidence to answer this."
            
        print(f"\n ✅ [VECTOR RESULT] (Confidence Score: {score:.2f})")
        print("-" * 70)
        print(f" 📍 Section: {best_chunk['section_header']}")
        
        clean_text = best_chunk['text'].replace('\n', ' ')
        wrapped_text = textwrap.fill(clean_text, width=68, initial_indent="    ", subsequent_indent="    ")
        
        if len(wrapped_text) > 500:
            wrapped_text = wrapped_text[:500] + "... [truncated]"
            
        print(f" 📄 Text:\n{wrapped_text}")
        print("-" * 70)
        return best_chunk['text']

if __name__ == "__main__":
    rag = SmartHybridRAG()
    
    rag.route_and_answer("What does location A4 represent in the test rig?")
    rag.route_and_answer("Which location code corresponds to the Bearing Hard Support?")
    rag.route_and_answer("What did the CT scan reveal about the press fit interface?")
    rag.route_and_answer("What is the exact measured hardness value of the failed spindle?")
    rag.route_and_answer("What is the material of the turbine blade?")