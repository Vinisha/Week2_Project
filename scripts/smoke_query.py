"""End-to-end smoke test: retrieve -> rerank -> Claude (with citations),
applying the same escalation gates as the API. Prints answers + citations."""
from app.config import settings
from app.generate import answer
from app.retrieve import hybrid_search

QUESTIONS = [
    "How long do refunds take?",
    "Do you ship internationally?",
    "My order is 8 days late, can I get the shipping fee back?",
    "What is the CEO's personal phone number?",  # out of scope -> should escalate
]

for q in QUESTIONS:
    print("\n" + "=" * 72)
    print("Q:", q)
    retrieval = hybrid_search(q)
    score = retrieval["top_rerank_score"]
    print(f"top_rerank_score={score:.3f}  candidates={len(retrieval['results'])}")

    if not retrieval["results"] or score < settings.min_rerank_score:
        print("-> ESCALATE (low retrieval confidence)")
        continue

    out = answer(q, retrieval["results"])
    if not out["answered"]:
        print("-> ESCALATE (model could not ground an answer)")
        print("   model said:", out["answer"][:160])
        continue

    print("A:", out["answer"])
    for c in out["citations"]:
        snippet = (c.get("cited_text") or "")[:80]
        print(f"   [cite] {c.get('doc_id')}: {snippet!r}")
