import json
from pathlib import Path
from ingestion.csv_parser import ingest_csv

def run_parser_eval():
    csv_path = Path("data/documents.csv")
    labels = json.loads(Path("data/parser_labels.json").read_text())
    if not csv_path.exists(): return print("SKIP: CSV not found")
    
    docs = {d["doc_id"]: d for d in ingest_csv(csv_path)}
    correct, total = 0, 0
    
    for entry in labels:
        parsed = docs.get(entry["doc_id"])
        if not parsed: continue
        for field, exp in entry["expected"].items():
            total += 1
            if parsed.get(field) == exp: correct += 1
            else: print(f"Mismatch {entry['doc_id']}.{field}: expected {exp}, got {parsed.get(field)}")
            
    print(f"Parser field accuracy: {correct}/{total} = {correct/total:.1%}" if total else "No labels.")

if __name__ == "__main__":
    run_parser_eval()