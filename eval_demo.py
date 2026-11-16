import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import EngineeringRAG
from access.rules import UserContext

def run_demo_eval():
    print("="*80)
    print("🚀 INITIALIZING HYBRID ARCHITECTURE EVALUATION")
    print("="*80)
    
    rag = EngineeringRAG()
    engineer = UserContext("u1", "engineering", frozenset(["combustion", "structures"]))
    
    test_cases = [
        {
            "category": "1. SYMBOLIC (SQL) ROUTE",
            "query": "What does location A4 represent in the test rig?",
            "expected": "Should route to SQL and return 'Damper Soft Support'."
        },
        {
            "category": "2. NEURAL (VECTOR) ROUTE",
            "query": "Why did the spindle shaft fail during the test?",
            "expected": "Should route to Vector and pull the Conclusion text."
        },
        {
            "category": "3. HYBRID ROUTE",
            "query": "What is the required hardness value and why did it cause failure?",
            "expected": "Should route to HYBRID (or fallback) and combine table/paragraph context."
        },
        {
            "category": "4. LATE-BINDING VQA (IMAGES)",
            "query": "Look at the Campbell diagram. What is the critical speed?",
            "expected": "Should find the image chunk, run VQA, and analyze the diagram."
        },
        {
            "category": "5. SECURITY & ACL",
            "query": "What is the budget for the test rig?",
            "expected": "Should trigger the Hard Blocklist and refuse."
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}: {test['category']}")
        print(f"❓ QUERY: {test['query']}")
        print(f"📋 EXPECTED: {test['expected']}")
        print(f"{'-'*80}")
        
        # Run the RAG pipeline
        res = rag.answer(test['query'], engineer)
        
        if res['refused']:
            print(f"🚫 REFUSED: {res['answer']}")
        else:
            print(f"🤖 ANSWER:\n{res['answer']}")
            print(f"\n📚 SOURCES: {[s['doc_id'] for s in res['sources']]}")
            for s in res['sources']:
                if 'image_path' in s:
                    print(f"🖼️  IMAGE ATTACHED: {s['image_path']}")
        
        print(f"{'='*80}")

if __name__ == "__main__":
    run_demo_eval()