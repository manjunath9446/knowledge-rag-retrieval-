import sqlite3
import pandas as pd

class HybridRouter:
    def __init__(self, db_path="hycosys_tables.db"):
        self.db_path = db_path

    def execute_sql_query(self, query: str):
        """In production, an LLM generates the SQL. Here we use a simple rule."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Available SQL Tables: {tables}")
        
        # Example query logic based on the TEI paper
        if "hardness" in query.lower():
            try:
                # Fetch the table that has hardness data
                df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)
                conn.close()
                return f"SQL Result (Table): \n{df.to_string()}"
            except Exception as e:
                return str(e)
                
        conn.close()
        return "No specific numeric table matched."

    def route(self, user_query: str):
        print(f"\n--- Routing Query: '{user_query}' ---")
        
        # Simple deterministic routing rule
        if any(word in user_query.lower() for word in ["value", "how much", "number", "table", "hardness", "stress"]):
            print("-> Routing to SYMBOLIC (SQL) layer.")
            return self.execute_sql_query(user_query)
        else:
            print("-> Routing to NEURAL (Vector RAG) layer.")
            return "Proceed with BGE Embeddings + BM25."

if __name__ == "__main__":
    router = HybridRouter()
    router.route("What is the required hardness value?")
    router.route("What caused the spindle shaft failure?")