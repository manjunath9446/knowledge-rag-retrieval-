import os
import json
import asyncio
import re
import sqlite3
import pandas as pd
from pathlib import Path
from io import StringIO

from llama_cloud import AsyncLlamaCloud

# ==========================================
# SETUP & PATHS
# ==========================================
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-bloeydSW7DOu5z1gevgbcNsRojWDd5G5edVXpgaa59kQyVq2"

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

PDF_FILE = SCRIPT_DIR / "GT2026-178735.pdf"
if not PDF_FILE.exists():
    PDF_FILE = PROJECT_ROOT / "GT2026-178735.pdf"

DB_FILE = SCRIPT_DIR / "hycosys_tables.db"
CHUNKS_FILE = PROJECT_ROOT / "data" / "parsed_chunks.json"

CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

def extract_markdown_text(md_obj) -> str:
    if isinstance(md_obj, str):
        return md_obj
    if hasattr(md_obj, "pages"):
        return "\n\n".join([getattr(p, "markdown", getattr(p, "text", str(p))) for p in md_obj.pages])
    if hasattr(md_obj, "text"):
        return md_obj.text
    return str(md_obj)

async def parse_and_chunk():
    print(f"[1/3] Uploading and Parsing {PDF_FILE.name} with LlamaCloud v2...")
    
    client = AsyncLlamaCloud(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
        base_url="https://api.cloud.llamaindex.ai"
    )
    
    try:
        # 1. UPLOAD THE FILE
        with open(PDF_FILE, "rb") as f:
            file_obj = await client.files.create(file=f, purpose="parse")
            
        print(f"      File uploaded. ID: {file_obj.id}")
        
        # 2. START THE PARSING JOB
        print("      Starting parse job...")
        job = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            expand=["markdown"]
        )
        
        # 3. POLL FOR COMPLETION
        print("      Waiting for parsing to complete...")
        while job.job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
            await asyncio.sleep(3)
            job = await client.parsing.get(job.job.id, expand=["markdown"])
            
        if job.job.status == "FAILED":
            print("      [ERROR] LlamaParse job failed.")
            return
            
        # 4. FETCH THE MARKDOWN RESULT
        full_markdown = extract_markdown_text(job.markdown)
        print("[1/3] Parsing complete.")

        # ==========================================
        # 2. SYMBOLIC LAYER: Extract Tables to SQLite
        # ==========================================
        print("[2/3] Extracting tables to SQLite for exact querying...")
        conn = sqlite3.connect(DB_FILE)
        
        # DIAGNOSTIC: Print a sample of the markdown to see how tables look
        print(f"      [DEBUG] Markdown sample:\n{full_markdown[:500]}...\n")
        
        # 1. Extract HTML tables (LlamaParse often uses these for complex layouts)
        html_tables = re.findall(r"(<table.*?</table>)", full_markdown, re.DOTALL | re.IGNORECASE)
        
        # 2. Extract Markdown tables (fallback)
        md_tables = re.findall(r"((?:^\|.*\|$\n?)+)", full_markdown, re.MULTILINE)
        
        all_tables = html_tables + md_tables
        table_count = 0
        
        for idx, table_data in enumerate(all_tables):
            try:
                # Check if it's HTML or Markdown
                if table_data.strip().startswith("<table"):
                    # Parse HTML table
                    dfs = pd.read_html(StringIO(table_data))
                    df = dfs[0]
                else:
                    # Parse Markdown table
                    df = pd.read_csv(StringIO(table_data), sep="|", skipinitialspace=True)
                    df = df.dropna(axis=1, how='all')
                    df.columns = [c.strip() for c in df.columns if c.strip() != ""]
                
                if not df.empty:
                    table_name = f"table_{table_count+1}"
                    df.to_sql(table_name, conn, if_exists="replace", index=False)
                    print(f"   -> Extracted {table_name} ({len(df)} rows, {len(df.columns)} cols)")
                    table_count += 1
            except Exception as e:
                print(f"   -> Skipped malformed table {idx+1}: {e}")
                
        conn.close()

        # ==========================================
        # 3. STRUCTURE-AWARE CHUNKING
        # ==========================================
        print("[3/3] Applying Structure-Aware Chunking...")
        chunks = []
        
        # Split by Markdown Headers
        sections = re.split(r"\n(?=#{1,3} )", full_markdown)
        
        for sec in sections:
            if not sec.strip(): continue
            
            header_match = re.match(r"^(#{1,3} .*)", sec)
            header = header_match.group(1) if header_match else "General"
            
            chunks.append({
                "chunk_id": f"chunk_{len(chunks)}",
                "section_header": header,
                "text": sec.strip(),
                "doc_id": "GT2026-178735"
            })
            
        with open(CHUNKS_FILE, "w") as f:
            json.dump(chunks, f, indent=2)

        print(f"\n[SUCCESS] Created {len(chunks)} structure-aware chunks.")
        print(f"[SUCCESS] Extracted {table_count} tables to {DB_FILE}.")
        print(f"[SUCCESS] Saved text chunks to {CHUNKS_FILE}.")

    except Exception as e:
        print(f"[ERROR] Failed to parse: {e}")

if __name__ == "__main__":
    asyncio.run(parse_and_chunk())