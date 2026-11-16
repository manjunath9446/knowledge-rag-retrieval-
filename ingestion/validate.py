def sanitize_and_validate(docs: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Cleans messy data instead of crashing. 
    Returns a tuple of (cleaned_docs, list_of_warnings).
    """
    warnings = []
    clean_docs = []
    seen_ids = set()

    # Pass 1: Type coercion and De-duplication
    for d in docs:
        # 1. Fix string "null"
        if d.get("supersedes") in ["null", "None", "none", ""]:
            d["supersedes"] = None
            
        # 2. Fix string booleans (e.g. "true" -> True)
        if isinstance(d.get("approved"), str):
            d["approved"] = d["approved"].lower() == "true"
            
        # 3. Handle duplicate IDs (Overwrite the old one with the new one)
        if d["doc_id"] in seen_ids:
            warnings.append(f"Duplicate ID '{d['doc_id']}' found. Overwriting previous entry.")
            clean_docs = [x for x in clean_docs if x["doc_id"] != d["doc_id"]]
            
        seen_ids.add(d["doc_id"])
        clean_docs.append(d)

    # Pass 2: Fix Referential Integrity (Broken Supersedes links)
    valid_ids = {d["doc_id"] for d in clean_docs}
    for d in clean_docs:
        sup = d.get("supersedes")
        if sup and sup not in valid_ids:
            warnings.append(f"{d['doc_id']} tried to supersede '{sup}', but it doesn't exist. Link removed.")
            d["supersedes"] = None

    # Pass 3: Strict Validation (Fail fast on critical schema errors)
    for d in clean_docs:
        if not isinstance(d.get("approved"), bool):
            raise ValueError(f"{d['doc_id']}: 'approved' must be a boolean.")
        if not isinstance(d.get("revision"), str):
            raise ValueError(f"{d['doc_id']}: 'revision' must be a string.")
            
    return clean_docs, warnings