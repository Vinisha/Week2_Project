"""Answer generation with Claude using the citations feature.

Each retrieved chunk is passed as a plain-text document with citations enabled,
so Claude attributes every claim to a specific source sentence. We treat the
*presence of citations* as the signal that Claude actually answered from the
provided documents; an answer with zero citations is treated as an abstention
and routed to a human."""
import json

import anthropic

from .config import settings

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = (
    "You are a customer support assistant for an e-commerce product. "
    "Answer the customer's question using ONLY the provided documents "
    "(FAQs, product manuals, and support tickets).\n"
    "Rules:\n"
    "- Ground every factual statement in the documents and cite the source.\n"
    "- Be concise, friendly, and direct.\n"
    "- Do not invent policies, prices, order details, or steps not in the documents.\n"
    "- If the documents do not contain enough information to answer, reply with "
    'exactly: "I don\'t have enough information to answer that."'
)

ABSTAIN_MARKER = "i don't have enough information"


def _build_documents(chunks: list[dict]) -> list[dict]:
    blocks = []
    for c in chunks:
        updated = c.get("updated_at")
        context = json.dumps(
            {
                "doc_id": c["doc_id"],
                "source_type": c["source_type"],
                "updated_at": updated.isoformat() if updated else None,
            }
        )
        blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": c["content"],
                },
                "title": (c.get("title") or c["doc_id"])[:200],
                "context": context,  # metadata for the model; not citable
                "citations": {"enabled": True},
            }
        )
    return blocks


def answer(question: str, chunks: list[dict]) -> dict:
    content = _build_documents(chunks)
    content.append({"type": "text", "text": question})

    resp = _client.messages.create(
        model=settings.gen_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    text_parts: list[str] = []
    citations: list[dict] = []
    for block in resp.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for cit in getattr(block, "citations", None) or []:
            idx = getattr(cit, "document_index", -1)
            src = chunks[idx] if 0 <= idx < len(chunks) else {}
            citations.append(
                {
                    "doc_id": src.get("doc_id"),
                    "source_type": src.get("source_type"),
                    "title": getattr(cit, "document_title", None),
                    "cited_text": getattr(cit, "cited_text", None),
                }
            )

    answer_text = "".join(text_parts).strip()
    answered = bool(citations) and ABSTAIN_MARKER not in answer_text.lower()
    return {"answer": answer_text, "citations": citations, "answered": answered}
