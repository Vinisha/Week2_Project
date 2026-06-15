"""Dedicated reranker stage.

A cross-encoder scores each (query, chunk) pair *jointly*, which is far more
accurate at judging relevance than the first-stage retrieval scores (dense
cosine / sparse BM25 each look at the query and chunk separately). We run it on
the fused candidate set and keep the best `top_k` for generation.

Scores are sigmoid-normalized to 0-1 so they're usable as an interpretable
confidence gate (`MIN_RERANK_SCORE`).

Default backend is a local cross-encoder (no API key, runs offline). Swap
`RERANKER_MODEL` for any HuggingFace cross-encoder, or replace `_score()` with
a hosted reranker (Cohere/Voyage) if you prefer."""
import math
from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so the rest of the app doesn't require torch.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def _score(query: str, contents: list[str]) -> list[float]:
    pairs = [(query, c) for c in contents]
    raw = _model().predict(pairs)  # relevance logits
    return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]  # -> 0..1


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    scores = _score(query, [c["content"] for c in candidates])
    ranked = []
    for cand, score in zip(candidates, scores):
        row = dict(cand)
        row["rerank_score"] = score
        ranked.append(row)
    ranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return ranked[:top_k]
