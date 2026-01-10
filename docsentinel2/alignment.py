import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticAligner:
    def __init__(self, embed_model):
        self.embed_model = embed_model

    def __call__(self, old_sentences, new_sentences, sim_threshold=0.75):
        if not old_sentences and not new_sentences:
            return []

        old_emb = self.embed_model.encode(old_sentences, convert_to_numpy=True)
        new_emb = self.embed_model.encode(new_sentences, convert_to_numpy=True)

        n, m = len(old_sentences), len(new_sentences)
        sim_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                sim_matrix[i, j] = (
                    np.dot(old_emb[i], new_emb[j]) /
                    (np.linalg.norm(old_emb[i]) * np.linalg.norm(new_emb[j]) + 1e-10)
                )

        # Needleman-Wunsch with gap penalty
        gap_penalty = -0.4
        dp = np.zeros((n + 1, m + 1))

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + gap_penalty
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + gap_penalty

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = dp[i - 1][j - 1] + sim_matrix[i - 1][j - 1]
                delete = dp[i - 1][j] + gap_penalty
                insert = dp[i][j - 1] + gap_penalty
                dp[i][j] = max(match, delete, insert)

        i, j = n, m
        alignment = []

        while i > 0 or j > 0:
            if i > 0 and j > 0 and \
                dp[i][j] == dp[i - 1][j - 1] + sim_matrix[i - 1][j - 1]:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
                alignment.append((i - 1, None))  # sentence deleted
                i -= 1
            else:
                alignment.append((None, j - 1))  # sentence added
                j -= 1

        alignment.reverse()
        return alignment
