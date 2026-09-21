"""Sentence-embedding retrieval: the one model shared by the coarse pass
(over candidates.py's merged-segment windows) and the fine pass (over
span.py's word-level sub-windows).

Model loading is intentionally the only place torch/sentence-transformers
get imported in this module, and only inside functions -- so
``import solution.retrieval`` stays cheap and this module is safe to import
from unit tests that never actually load a model.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from solution.config import RetrievalConfig


class Retriever:
    """Thin wrapper around a sentence-transformers bi-encoder.

    BAAI/bge-* models are trained with asymmetric query/passage prefixes
    (see ``query_prefix``/``passage_prefix`` in config) -- using them
    measurably improves retrieval quality for this family and costs
    nothing, so we apply them here rather than leaving it to call sites to
    remember.
    """

    def __init__(self, config: RetrievalConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._model = SentenceTransformer(config.embedding_model, device=config.device)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"{self.config.query_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"{self.config.passage_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,   # so cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def top_k(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """Indices and cosine-similarity scores of the top-k candidates.

    Assumes both inputs are already L2-normalised (Retriever._encode does
    this), so this is a plain dot product, not a full cosine-similarity
    computation -- cheap enough to call once per question with no caching
    beyond what the caller already has.
    """

    if candidate_vectors.shape[0] == 0:
        return []

    scores = candidate_vectors @ query_vector
    k = min(k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx]
