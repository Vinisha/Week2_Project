"""Two-stage retrieval:
  1. Hybrid recall: dense (pgvector cosine) + sparse (Postgres full-text),
     fused with Reciprocal Rank Fusion.
  2. Rerank: a cross-encoder re-scores the fused candidates and keeps top_k.

Stale support tickets are filtered at query time as a safety net on top of the
ingestion-time rule.

`top_rerank_score` (best cross-encoder relevance, 0-1) is the primary confidence
signal for the escalation gate; `max_vscore` is kept for observability."""
import numpy as np

from .config import settings
from .db import get_conn
from .embeddings import embed_query
from .reranker import rerank

_RRF_K = 60


def _filter_clause() -> str:
    # Exclude support tickets older than STALE_DAYS.
    return (
        "NOT (source_type = 'support_ticket' "
        "AND updated_at < now() - make_interval(days => %s))"
    )


def hybrid_search(question: str) -> dict:
    # numpy array so pgvector's adapter sends a `vector` (not a float[]).
    qvec = np.asarray(embed_query(question), dtype=np.float32)
    sql_filter = _filter_clause()

    with get_conn() as conn, conn.cursor() as cur:
        # Dense candidates (cosine similarity = 1 - cosine distance).
        cur.execute(
            f"""
            SELECT id, doc_id, source_type, title, content, updated_at,
                   1 - (embedding <=> %s) AS vscore
            FROM chunks
            WHERE {sql_filter}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, settings.stale_days, qvec, settings.candidate_k),
        )
        dense = cur.fetchall()

        # Sparse candidates (full-text BM25-style ranking).
        cur.execute(
            f"""
            SELECT id, doc_id, source_type, title, content, updated_at,
                   ts_rank(tsv, plainto_tsquery('english', %s)) AS kscore
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
              AND {sql_filter}
            ORDER BY kscore DESC
            LIMIT %s
            """,
            (question, question, settings.stale_days, settings.candidate_k),
        )
        sparse = cur.fetchall()

    # Stage 1: fuse recall lists; keep a wide candidate set for the reranker.
    fused = _rrf(dense, sparse)[: settings.rerank_candidates]
    # Stage 2: cross-encoder rerank down to top_k.
    reranked = rerank(question, fused, settings.top_k)

    top_rerank_score = reranked[0]["rerank_score"] if reranked else 0.0
    max_vscore = max((r["vscore"] for r in dense), default=0.0)
    return {
        "results": reranked,
        "top_rerank_score": float(top_rerank_score),
        "max_vscore": float(max_vscore),
    }


def _rrf(dense: list[dict], sparse: list[dict]) -> list[dict]:
    scores: dict[int, float] = {}
    meta: dict[int, dict] = {}
    for ranked in (dense, sparse):
        for rank, row in enumerate(ranked):
            cid = row["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            meta[cid] = row
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    out = []
    for cid in order:
        row = dict(meta[cid])
        row["fused_score"] = scores[cid]
        out.append(row)
    return out
