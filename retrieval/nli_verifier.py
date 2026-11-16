from sentence_transformers import CrossEncoder
import numpy as np

class NLIVerifier:
    def __init__(self):
        print("[INFO] Loading NLI Verifier (DeBERTa)...")
        self.model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")

    def verify(self, premise: str, hypothesis: str) -> float:
        scores = self.model.predict([(premise, hypothesis)])
        # Output is shape (1, 3): [contradiction, neutral, entailment]
        # Softmax calculation across axis 1
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        return float(probs[0][2]) # Return the 'entailment' probability