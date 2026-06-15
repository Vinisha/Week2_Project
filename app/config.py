"""Central configuration, loaded from environment / .env."""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


class Settings:
    # Postgres
    database_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/ragbot")

    # Nebius embeddings (OpenAI-compatible)
    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    nebius_base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")
    embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    embed_dim = _int("EMBED_DIM", "768")
    embed_query_prefix = os.getenv("EMBED_QUERY_PREFIX", "")
    embed_doc_prefix = os.getenv("EMBED_DOC_PREFIX", "")

    # Claude generation
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    gen_model = os.getenv("GEN_MODEL", "claude-sonnet-4-6")

    # Pipeline tuning
    chunk_max_tokens = _int("CHUNK_MAX_TOKENS", "512")
    top_k = _int("TOP_K", "5")
    candidate_k = _int("CANDIDATE_K", "20")
    # Escalation gate: best embedding cosine similarity must clear this, else we
    # hand off to a human. Embedding similarity separates relevant from
    # out-of-scope more reliably than the cross-encoder for paraphrased questions.
    min_vscore = _float("MIN_VSCORE", "0.50")
    stale_days = _int("STALE_DAYS", "30")

    # Reranker (cross-encoder): reorders the retrieved context before generation.
    # Not used for the escalation gate (see min_vscore).
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates = _int("RERANK_CANDIDATES", "20")   # how many fused hits to rerank

    # Human handoff
    escalation_email = os.getenv("ESCALATION_EMAIL", "vinisha6789@gmail.com")
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = _int("SMTP_PORT", "587")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    email_from = os.getenv("EMAIL_FROM") or os.getenv("SMTP_USER", "")


settings = Settings()

# Valid document categories. Only support tickets are subject to the staleness rule.
SOURCE_TYPES = {"faq", "manual", "support_ticket"}
