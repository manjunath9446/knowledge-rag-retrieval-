import json
from pathlib import Path
from ingestion.validate import sanitize_and_validate
from ingestion.csv_parser import ingest_csv

def load_documents(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".json":
        raw_docs = json.loads(p.read_text())
    elif p.suffix == ".csv":
        raw_docs = ingest_csv(p)
    else:
        raise ValueError("Unsupported format")
        
    # Sanitize instead of hard validate
    docs, warnings = sanitize_and_validate(raw_docs)
    for w in warnings:
        print(f"[QA/QC WARNING] {w}")
        
    return docs