"""Embeddings via Nebius Token Factory (OpenAI-compatible endpoint)."""
from openai import OpenAI

from .config import settings

_client = OpenAI(api_key=settings.nebius_api_key, base_url=settings.nebius_base_url)


def _embed(texts: list[str], prefix: str) -> list[list[float]]:
    if not texts:
        return []
    inputs = [f"{prefix}{t}" if prefix else t for t in texts]
    out: list[list[float]] = []
    for i in range(0, len(inputs), 64):  # batch to stay under request limits
        batch = inputs[i : i + 64]
        # `dimensions` truncates via Matryoshka (Qwen3-Embedding default is 4096;
        # we use EMBED_DIM so vectors stay within pgvector's 2000-dim index limit).
        resp = _client.embeddings.create(
            model=settings.embed_model, input=batch, dimensions=settings.embed_dim
        )
        for d in resp.data:
            v = d.embedding
            if len(v) != settings.embed_dim:
                raise ValueError(
                    f"Embedding dim {len(v)} != configured EMBED_DIM "
                    f"{settings.embed_dim}. Update EMBED_MODEL/EMBED_DIM to match."
                )
            out.append(v)
    return out


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, settings.embed_doc_prefix)


def embed_query(text: str) -> list[float]:
    return _embed([text], settings.embed_query_prefix)[0]
