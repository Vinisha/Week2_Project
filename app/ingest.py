"""Indexing pipeline: clean -> semantic chunk -> embed -> upsert into pgvector.

Idempotent on doc_id (re-indexing replaces a document's chunks). Skips
unchanged content via a content hash. Enforces the staleness rule for
support tickets only (FAQs and manuals are evergreen)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from .chunking import semantic_chunk
from .cleaning import clean_text
from .config import SOURCE_TYPES, settings
from .db import get_conn
from .embeddings import embed_documents


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def remove_document(doc_id: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        conn.commit()


def _is_stale(source_type: str, updated_at: datetime) -> bool:
    if source_type != "support_ticket":
        return False
    age_days = (datetime.now(timezone.utc) - updated_at).days
    return age_days > settings.stale_days


def ingest_document(
    doc_id: str,
    source_type: str,
    raw_text: str,
    title: str | None = None,
    source_uri: str | None = None,
    updated_at: datetime | None = None,
) -> dict:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
    updated_at = updated_at or datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    # Staleness rule (support tickets only): drop and skip.
    if _is_stale(source_type, updated_at):
        remove_document(doc_id)
        return {"doc_id": doc_id, "skipped": "stale", "chunks": 0}

    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        return {"doc_id": doc_id, "skipped": "empty", "chunks": 0}

    content_hash = _hash(cleaned)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
        if row and row["content_hash"] == content_hash:
            return {"doc_id": doc_id, "skipped": "unchanged", "chunks": 0}

    chunks = semantic_chunk(cleaned, max_tokens=settings.chunk_max_tokens)
    if not chunks:
        return {"doc_id": doc_id, "skipped": "no_chunks", "chunks": 0}

    vectors = embed_documents(chunks)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (doc_id, source_type, title, source_uri, content_hash, updated_at, indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (doc_id) DO UPDATE SET
                source_type  = EXCLUDED.source_type,
                title        = EXCLUDED.title,
                source_uri   = EXCLUDED.source_uri,
                content_hash = EXCLUDED.content_hash,
                updated_at   = EXCLUDED.updated_at,
                indexed_at   = now()
            """,
            (doc_id, source_type, title, source_uri, content_hash, updated_at),
        )
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            cur.execute(
                """
                INSERT INTO chunks
                    (doc_id, chunk_index, source_type, title, content, updated_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (doc_id, idx, source_type, title, chunk, updated_at,
                 np.asarray(vec, dtype=np.float32)),
            )
        conn.commit()

    return {"doc_id": doc_id, "chunks": len(chunks)}
