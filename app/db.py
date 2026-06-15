"""Postgres + pgvector access. Connections yield dict rows and have the
vector type registered so Python lists/arrays adapt to the `vector` column."""
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from .config import settings


@contextmanager
def get_conn():
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            # The `vector` extension may not exist yet on a fresh DB (it's created
            # by init_db). Registration succeeds on every connection afterwards.
            pass
        yield conn


def init_db() -> None:
    """Create tables and indexes. Safe to run repeatedly."""
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        doc_id       text PRIMARY KEY,
        source_type  text NOT NULL,
        title        text,
        source_uri   text,
        content_hash text,
        updated_at   timestamptz NOT NULL DEFAULT now(),
        indexed_at   timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id          bigserial PRIMARY KEY,
        doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        chunk_index int  NOT NULL,
        source_type text NOT NULL,
        title       text,
        content     text NOT NULL,
        updated_at  timestamptz NOT NULL DEFAULT now(),
        embedding   vector({settings.embed_dim}) NOT NULL,
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    );

    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS chunks_tsv_idx
        ON chunks USING gin (tsv);
    CREATE INDEX IF NOT EXISTS chunks_type_updated_idx
        ON chunks (source_type, updated_at);

    CREATE TABLE IF NOT EXISTS escalations (
        id              bigserial PRIMARY KEY,
        question        text NOT NULL,
        conversation_id text,
        reason          text NOT NULL,
        retrieval_score double precision,
        emailed         boolean NOT NULL DEFAULT false,
        created_at      timestamptz NOT NULL DEFAULT now()
    );
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl)
        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
