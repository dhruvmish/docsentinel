from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences: List[str], convert_to_numpy=True):
        return self.model.encode(sentences, convert_to_numpy=convert_to_numpy)

    def similarity(self, s1: str, s2: str) -> float:
        emb = self.model.encode([s1, s2], convert_to_numpy=True)
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(sim)
