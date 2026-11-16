import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import EngineeringRAG
from agent.conversational import ConversationalAgent
from access.rules import UserContext
from ingestion.loaders import load_documents
from ingestion.validate import sanitize_and_validate

def run_adversarial_tests():
    print("="*80)
    print("INITIALIZING ENGINE & LOADING DATA")
    print("="*80)
    
    # Load base JSON and simulate the CSV upload
    base_docs = load_documents("data/documents.json")
    
    # We simulate the messy CSV data being loaded dynamically
    csv_docs = [
        {
            "doc_id": "SYS-2024-071", "title": "Fuel Valve Leak Test Results", "team": "systems",
            "access_level": "engineering", "date": "2024-08-09", "author": "Thabo Nkosi",
            "doc_type": "report", "content": "Leak test results for the hydrogen fuel shutoff valve assembly, measured external leakage of 0.02 sccm helium at 70 bar supply pressure, well within the 0.5 sccm allowable.",
            "revision": "A", "supersedes": None, "approved": True
        }
    ]
    
    combined_docs = base_docs + csv_docs
    rag = EngineeringRAG(data_source=combined_docs)
    
    eng_user = UserContext("u1", "engineering", frozenset(["combustion", "structures"]))
    lead_user = UserContext("u2", "leadership")
    
    def print_result(query, res, expected_behavior):
        print(f"\n❓ QUERY: {query}")
        print(f"📋 EXPECTED: {expected_behavior}")
        print(f"🤖 SYSTEM ANSWER: {res['answer']}")
        print(f"📚 SOURCES: {[s['doc_id'] for s in res['sources']]}")
        print(f"🚦 REFUSED: {res['refused']}")
        print("-" * 80)

    print("\n" + "="*80)
    print("CATEGORY 1: PRECISION RETRIEVAL")
    print("="*80)
    q1 = "What is the peak von Mises stress at the bore of the turbine disk?"
    r1 = rag.answer(q1, eng_user)
    print_result(q1, r1, "Exact numeric value (745 MPa) from STR-2025-006 (Rev B), no hedging.")

    q2 = "What is the minimum coolant flow rate required for the generator?"
    r2 = rag.answer(q2, eng_user)
    print_result(q2, r2, "Exact value (8.5 L/min) from SYS-2025-016.")

    print("\n" + "="*80)
    print("CATEGORY 2: MULTI-HOP SYNTHESIS")
    print("="*80)
    q3 = "What caused the swirler vane erosion and what did the metallurgical investigation find?"
    r3 = rag.answer(q3, eng_user)
    print_result(q3, r3, "Synthesizes COMB-2025-012 (degradation) and MAT-2025-030 (coating spallation).")

    print("\n" + "="*80)
    print("CATEGORY 3: REVISION & STATE MANAGEMENT")
    print("="*80)
    q4 = "What does the current in-work draft say about the turbine disk stress?"
    r4 = rag.answer(q4, eng_user)
    print_result(q4, r4, "Retrieves STR-2025-007 (Rev C Draft) and flags it as unapproved.")

    q5 = "What changed between Rev A and Rev B of the flashback margin assessment?"
    r5 = rag.answer(q5, eng_user)
    print_result(q5, r5, "Retrieves both COMB-2025-010 and COMB-2025-011, shows revision history.")

    print("\n" + "="*80)
    print("CATEGORY 4: SECURITY & ACCESS CONTROL")
    print("="*80)
    q6 = "What is the Series B funding allocation?"
    r6 = rag.answer(q6, eng_user)
    print_result(q6, r6, "Refuses (Engineering user asking for restricted Leadership data).")

    q7 = "Where is the budget authorization for the endurance test documented?"
    r7 = rag.answer(q7, eng_user)
    print_result(q7, r7, "Answers from SYS-2025-020 but scrubs [REDACTED] mention of LEAD-2025-002.")

    print("\n" + "="*80)
    print("CATEGORY 5: OUT-OF-DISTRIBUTION (OOD)")
    print("="*80)
    q8 = "What is the weather in Bengaluru today?"
    r8 = rag.answer(q8, eng_user)
    print_result(q8, r8, "Refuses (Low confidence / OOD).")

    print("\n" + "="*80)
    print("CATEGORY 6: AGENT MEMORY & PRONOUN RESOLUTION")
    print("="*80)
    agent = ConversationalAgent(rag)
    q9a = "What caused the flashback?"
    r9a = agent.ask(q9a, eng_user)
    print_result(q9a, r9a, "Retrieves TEST-2025-039 / CFD-2025-012.")
    
    q9b = "And what was the fix for it?"
    r9b = agent.ask(q9b, eng_user)
    print_result(q9b, r9b, "Resolves pronoun 'it' to flashback, retrieves DEC-2025-007.")

    print("\n" + "="*80)
    print("CATEGORY 7: COMPOUND CONSTRAINTS")
    print("="*80)
    q10 = "What is the bearing preload AND the gearbox lubrication flow rate?"
    r10 = rag.answer(q10, eng_user)
    print_result(q10, r10, "Answers both distinct facts (200N preload, 4.2 L/min oil flow).")

    print("\n" + "="*80)
    print("CATEGORY 8: NEGATIVE CONSTRAINTS (FALSE PREMISE)")
    print("="*80)
    q11 = "Why did we choose aluminum for the combustor liner?"
    r11 = rag.answer(q11, eng_user)
    print_result(q11, r11, "Corrects the premise (it's Hastelloy X), does not hallucinate a rationale for aluminum.")

if __name__ == "__main__":
    run_adversarial_tests()