"""Quick Nebius embeddings connectivity check.

Embeds one sentence and reports the vector dimension. Never prints the API key.
Confirms the key works AND that EMBED_DIM matches the model's real output.

Run:
    python scripts/test_nebius.py
"""
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NEBIUS_API_KEY")
base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
model = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
expected_dim = int(os.getenv("EMBED_DIM", "1024"))

if not api_key or api_key.startswith("your-"):
    sys.exit("ERROR: NEBIUS_API_KEY is not set in .env")

print(f"Endpoint : {base_url}")
print(f"Model    : {model}")

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    resp = client.embeddings.create(
        model=model, input="connectivity check", dimensions=expected_dim
    )
except Exception as exc:
    sys.exit(f"ERROR: request failed -> {exc}")

dim = len(resp.data[0].embedding)
print(f"Returned dimension: {dim}")

if dim != expected_dim:
    sys.exit(
        f"MISMATCH: model returns {dim} dims but EMBED_DIM={expected_dim}. "
        f"Set EMBED_DIM={dim} in .env before creating the pgvector tables."
    )

print(f"OK: embedding dim = {dim} (matches EMBED_DIM)")
