import csv
import re
from pathlib import Path

_REVISION_PAT = re.compile(r"\brev(?:ision)?\.?\s*([A-Za-z0-9]+)\b", re.IGNORECASE)
_SUPERSEDES_PAT = re.compile(r"\bsupersedes[:\s]+([A-Z]{2,}-\d{4}-\d{3})\b", re.IGNORECASE)
_DRAFT_PAT = re.compile(r"\b(DRAFT|UNAPPROVED|not\s+approved)\b", re.IGNORECASE)

def parse_row(row: dict) -> dict:
    content = row.get("content", "")
    
    # Strip rogue single/double quotes from the ends of the text
    content = content.strip(" '\"")
    title = row.get("title", "").strip(" '\"")
    
    sup_match = _SUPERSEDES_PAT.search(content)
    rev_match = _REVISION_PAT.search(content)
    
    return {
        "doc_id": row["doc_id"].strip(), 
        "title": title,
        "team": row.get("team", "unspecified").strip(), 
        "access_level": row.get("access_level", "engineering").strip(),
        "date": row.get("date", "").strip(), 
        "author": row.get("author", "").strip(), 
        "content": content,
        "revision": rev_match.group(1) if rev_match else "A",
        "supersedes": sup_match.group(1) if sup_match else None,
        "approved": not bool(_DRAFT_PAT.search(content)),
        "doc_type": row.get("doc_type", "report").strip(),
    }

def ingest_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [parse_row(row) for row in csv.DictReader(f)]