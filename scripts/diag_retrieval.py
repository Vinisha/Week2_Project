"""Diagnose retrieval/rerank scores per question (no LLM call)."""
from app.config import settings
from app.retrieve import hybrid_search

QUESTIONS = [
    # category 1 (exact) — these work
    "How many days do I have to return an item?",
    # category 2 (rephrased / relevant) — reported as wrongly escalating
    "Is there a delivery fee on a $30 order?",
    "Can I get my money back if the box is unopened?",
    "How do I make my account more secure?",
    "Where can I see where my package is?",
    # category 4 (out of scope) — must still escalate
    "What is the CEO's personal phone number?",
    "Can I pay with cryptocurrency?",
]

print(f"MIN_VSCORE gate = {settings.min_vscore}\n")
for q in QUESTIONS:
    r = hybrid_search(q)
    gate = "ANSWER" if r["results"] and r["max_vscore"] >= settings.min_vscore else "ESCALATE"
    print(f"[{gate}] top_rerank={r['top_rerank_score']:.3f}  max_vscore={r['max_vscore']:.3f}  | {q}")
    for hit in r["results"][:3]:
        snippet = hit["content"][:65].replace("\n", " ")
        print(f"        rerank={hit['rerank_score']:.3f}  {hit['doc_id']}: {snippet!r}")
    print()
