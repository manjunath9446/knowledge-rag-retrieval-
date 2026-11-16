import sqlite3
import json
import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "parsed_chunks.json"
DB_FILE = PROJECT_ROOT / "data" / "hycosys_tables.db"

def patch():
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
        
    conn = sqlite3.connect(DB_FILE)
    updated = 0
    
    for c in chunks:
        if "[Table Data:" in c.get("section_header", ""):
            # extract table name
            m = re.search(r"\[Table Data:\s*([^\]]+)\]", c["section_header"])
            if m:
                table_name = m.group(1).strip()
                try:
                    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                    
                    # Format as Markdown Table
                    drop_cols = ["doc_id"] if "doc_id" in df.columns else []
                    df_clean = df.drop(columns=drop_cols).fillna("")
                    
                    headers = " | ".join(str(col) for col in df_clean.columns)
                    separator = " | ".join(["---"] * len(df_clean.columns))
                    rows = [" | ".join(str(val) for val in row) for _, row in df_clean.iterrows()]
                    
                    md_table = f"**Table Name:** `{table_name}`\n\n| {headers} |\n| {separator} |\n" + "\n".join(f"| {row} |" for row in rows)
                    
                    c["text"] = md_table
                    updated += 1
                except Exception as e:
                    print(f"Error {table_name}: {e}")
                    
    conn.close()
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"✅ Patched {updated} table chunks into beautiful Markdown format.")

if __name__ == "__main__":
    patch()