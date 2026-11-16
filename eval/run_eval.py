from src.pipeline import EngineeringRAG
from access.rules import UserContext

def run_eval():
    rag = EngineeringRAG()
    eng_user = UserContext("u1", "engineering")
    lead_user = UserContext("u2", "leadership")
    
    cases = [
        ("What is the peak stress in the recuperator?", eng_user, False, []),
        ("What changed between Rev C and Rev D?", eng_user, False, []),
        ("What are the compensation bands?", eng_user, True, ["HR-2025-005", "People Ops"]),
        ("What are the compensation bands?", lead_user, False, []),
        ("Is there a funding plan for the AI tooling?", eng_user, False, ["BIZ-2025-002"]),
        ("What is the weather today?", eng_user, True, [])
    ]
    
    print(f"{'QUERY':<45} | {'REFUSED':<7} | {'LEAKS':<5} | {'STALE REV':<9}")
    print("-" * 75)
    
    leaks = 0
    stale_revs = 0
    superseded = {d["supersedes"] for d in rag.chunks if d["approved"] and d["supersedes"]}
    
    for query, user, expect_refusal, forbidden in cases:
        res = rag.answer(query, user)
        
        leak = any(term in res["answer"] for term in forbidden)
        if leak: leaks += 1
        
        is_history = "changed between" in query.lower()
        cited = {s["doc_id"] for s in res["sources"]}
        stale = cited & superseded if not is_history else set()
        if stale: stale_revs += 1
        
        print(f"{query[:43]:<45} | {str(res['refused']):<7} | {str(leak):<5} | {str(len(stale)):<9}")

    print("-" * 75)
    print(f"LEAKAGE: {leaks}/6 cases (PASS)" if leaks == 0 else f"LEAKAGE: {leaks} FAILURES")
    print(f"REVISION_CORRECTNESS: {'PASS' if stale_revs == 0 else 'FAIL'}")

if __name__ == "__main__":
    run_eval()