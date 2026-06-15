# RAG Support Bot

A retrieval-augmented chatbot that answers customer questions from **FAQs,
product manuals, and support tickets**, with grounded citations and automatic
hand-off to a human when it can't answer confidently.

## Stack

| Layer | Choice |
|---|---|
| Embeddings | Nebius Token Factory (OpenAI-compatible), 768-dim |
| Vector DB | Postgres + **pgvector** (vectors + metadata together) |
| Retrieval | Hybrid: dense (cosine) + sparse (full-text), fused with RRF |
| Reranker | Cross-encoder (`ms-marco-MiniLM`), local, re-scores candidates before generation |
| Generation | Claude `claude-sonnet-4-6` with the **citations** feature |
| Eval | RAGAS (Claude as judge): faithfulness, relevancy, context precision/recall |
| Chunking | Semantic, 512-token budget |
| Freshness | Webhook re-index; support tickets older than 30 days are dropped |
| Hand-off | Email to the support inbox **+** an `escalations` row for reconciliation |

## How it works

```
INDEX:  file/text --> clean --> semantic chunk --> embed (Nebius) --> pgvector
SERVE:  question --> embed --> hybrid retrieve --> rerank (cross-encoder)
                  --> gate 1 (rerank score floor)
                  --> Claude + citations --> gate 2 (must cite, else abstain)
                  --> answer + sources   OR   escalate (email + DB row)
```

The escalation gate has two checks:
1. **Retrieval confidence** — if the top **rerank** score is below
   `MIN_RERANK_SCORE`, escalate without even calling the LLM. (Rerank scores are
   more reliable than raw cosine, so they drive the gate.)
2. **Model grounding** — if Claude returns no citations (or the abstain phrase),
   we treat it as "couldn't answer" and escalate.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in keys + DATABASE_URL
```

You need a Postgres with the `vector` extension available (local Docker, or a
managed Postgres like Neon/Supabase/RDS). Then:

```bash
python -m app.db          # create tables + indexes
python run_ingest.py data # ingest the sample data/ folder
```

## Run

```bash
uvicorn app.api:app --reload
```

Ask a question:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "How long do refunds take?"}' | python -m json.tool
```

Re-index a document from your source system's webhook:

```bash
curl -s localhost:8000/webhook/reindex -H 'content-type: application/json' \
  -d '{"doc_id":"faqs/returns_policy.txt","source_type":"faq","path":"data/faqs/returns_policy.txt"}'
```

Remove a deleted document:

```bash
curl -s localhost:8000/webhook/reindex -H 'content-type: application/json' \
  -d '{"doc_id":"faqs/old.txt","source_type":"faq","deleted":true}'
```

## Configuration notes

- **Embedding model / dimension:** `EMBED_MODEL` must be a 768-dim model in your
  Nebius catalog, and `EMBED_DIM` must match it. A dimension mismatch fails loudly
  at embed time. Some models (E5/BGE) retrieve better with query/passage prefixes —
  set `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` if so.
- **Escalation email:** without SMTP configured the bot still records the
  `escalations` row (source of truth); email is best-effort.
- **Staleness:** only `support_ticket` documents expire (after `STALE_DAYS`).
  FAQs and manuals are evergreen and only change via the re-index webhook.

## Evaluation (RAGAS)

Measures the pipeline against your targets (95% faithfulness, 90% relevance)
using Claude as the judge and Nebius for embeddings.

```bash
pip install -r requirements-eval.txt
python -m eval.run_eval
```

It runs the real retrieve -> rerank -> generate path over `eval/eval_set.jsonl`
(question + ground_truth per line), scores the answered questions on
faithfulness / answer relevancy / context precision / context recall, prints
each metric against its target, and reports the escalation rate (out-of-scope
questions should escalate, not be answered).

**Tuning loop:** run eval -> adjust `MIN_RERANK_SCORE`, `TOP_K`,
`RERANK_CANDIDATES`, and the chunking params -> re-run. Grow `eval_set.jsonl`
to ~50–100 real Q&A pairs for meaningful numbers.

> RAGAS APIs shift between minor versions; `requirements-eval.txt` pins the
> 0.2.x line. On a different version you may need to tweak metric imports.

## Next steps (not yet built)

- Streaming responses for lower perceived latency.
- Per-answer 👍/👎 feedback logging to grow the eval set from real traffic.
