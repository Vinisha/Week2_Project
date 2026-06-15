"""Offline RAG evaluation with RAGAS.

Runs the live retrieve -> rerank -> generate path over eval_set.jsonl, then
scores the answered questions on:
  - faithfulness      (answer grounded in retrieved context)   target >= 0.95
  - answer relevancy  (answer addresses the question)          target >= 0.90
  - context precision (retrieved context is on-topic)
  - context recall    (retrieved context covers the reference)

Judge LLM = Claude (same model as the app); embeddings = Nebius. Questions the
pipeline escalates (no confident answer) are reported separately as an
escalation rate rather than scored.

NOTE: RAGAS APIs change between minor versions. This targets ragas 0.2.x
(see requirements-eval.txt). If you install a different version you may need to
adjust metric class names or imports.

Install: pip install -r requirements-eval.txt
Run:     python -m eval.run_eval
"""
import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.config import settings
from app.generate import answer
from app.retrieve import hybrid_search

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"
TARGETS = [("faithfulness", 0.95, "faith"), ("answer relevancy", 0.90, "relevanc")]


def build_samples():
    """Run the real pipeline per question; collect answered samples for scoring."""
    samples: list[SingleTurnSample] = []
    escalated: list[str] = []
    for line in EVAL_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        question = item["question"]

        retrieval = hybrid_search(question)
        contexts = [r["content"] for r in retrieval["results"]]
        if retrieval["results"] and retrieval["max_vscore"] >= settings.min_vscore:
            result = answer(question, retrieval["results"])
        else:
            result = {"answer": "", "answered": False}

        if not result["answered"]:
            escalated.append(question)
            continue

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=contexts,
                reference=item["ground_truth"],
            )
        )
    return samples, escalated


def main():
    samples, escalated = build_samples()
    total = len(samples) + len(escalated)
    print(f"Answered: {len(samples)}  |  Escalated: {len(escalated)}  |  Total: {total}")
    if not samples:
        print("No answered samples to score.")
        return

    judge = LangchainLLMWrapper(ChatAnthropic(model=settings.gen_model, max_tokens=1024))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            dimensions=settings.embed_dim,
            # Nebius rejects tiktoken token-id input ("Tokenized input is not
            # supported"); send raw strings instead.
            check_embedding_ctx_length=False,
        )
    )

    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics)

    df = result.to_pandas()
    base_cols = {"user_input", "response", "retrieved_contexts", "reference"}
    metric_cols = [c for c in df.columns if c not in base_cols]

    print("\n=== RAGAS scores (mean over answered questions) ===")
    means: dict[str, float] = {}
    for col in metric_cols:
        try:
            means[col] = float(df[col].mean())
            print(f"  {col:36s} {means[col]:.3f}")
        except Exception:
            pass

    print("\n=== vs targets ===")
    for label, target, key in TARGETS:
        match = next(((c, v) for c, v in means.items() if key in c), (None, None))
        _, value = match
        if value is None:
            print(f"  {label:18s} (metric not found)")
            continue
        flag = "PASS" if value >= target else "BELOW"
        print(f"  {label:18s} {value:.3f}  target {target:.2f}  [{flag}]")

    print(f"\n  escalation rate     {len(escalated)}/{total}")


if __name__ == "__main__":
    main()
