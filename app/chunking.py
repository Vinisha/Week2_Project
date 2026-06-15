"""Structure-aware chunking.

For FAQ / manual / ticket content the strongest topic boundary is *structural*
(paragraphs and headings), not sentence-embedding similarity: a short FAQ is
internally similar enough that semantic splitting keeps it as one blob, which
makes precise retrieval impossible (a paraphrased question gets scored against a
big multi-topic passage and the relevance washes out).

So we split on blank-line paragraphs, attach a heading to the paragraph it
introduces, and emit one focused chunk per paragraph/section. A block is only
sentence-packed when it exceeds the token budget (e.g. a long manual section)."""
import re

_TOK_PER_CHAR = 4  # ~4 chars/token, for sizing only


def _est_tokens(s: str) -> int:
    return max(1, len(s) // _TOK_PER_CHAR)


def _is_heading(block: str) -> bool:
    # Markdown heading, or a short standalone line that isn't a full sentence
    # (e.g. a section title like "Returns and Refunds").
    if block.startswith("#"):
        return True
    return (
        "\n" not in block
        and len(block) <= 60
        and not block.endswith((".", "!", "?", ":"))
    )


def _split_blocks(text: str) -> list[str]:
    raw = [b.strip() for b in re.split(r"\n{2,}", text.strip()) if b.strip()]
    blocks: list[str] = []
    pending = ""  # accumulates heading lines to prepend to the next real block
    for b in raw:
        if _is_heading(b):
            pending = f"{pending}\n{b}".strip() if pending else b
            continue
        if pending:
            b = f"{pending}\n{b}"
            pending = ""
        blocks.append(b)
    if pending:  # trailing heading with no body
        blocks.append(pending)
    return blocks


def _sentence_pack(block: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", block)
    chunks: list[str] = []
    cur: list[str] = []
    tok = 0
    for s in sentences:
        t = _est_tokens(s)
        if cur and tok + t > max_tokens:
            chunks.append(" ".join(cur))
            cur, tok = [], 0
        cur.append(s)
        tok += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _normalize(block: str) -> str:
    return re.sub(r"\s*\n\s*", " ", block).strip()  # flatten internal newlines


def semantic_chunk(text: str, max_tokens: int = 512, **_) -> list[str]:
    chunks: list[str] = []
    for block in _split_blocks(text):
        if _est_tokens(block) <= max_tokens:
            chunks.append(_normalize(block))
        else:
            chunks.extend(_normalize(c) for c in _sentence_pack(block, max_tokens))
    return [c for c in chunks if c]
