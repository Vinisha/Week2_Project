"""FastAPI service: customer query endpoint + webhook for re-indexing."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .cleaning import load_file
from .config import settings
from .escalate import record_and_email
from .generate import answer
from .ingest import ingest_document, remove_document
from .retrieve import hybrid_search

app = FastAPI(title="RAG Support Bot")

# Allow the page to call /query even when it's loaded from a different origin
# (file://, a preview panel, etc.), not just when served by this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_HTML = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the chat UI."""
    return _UI_HTML

HANDOFF_MESSAGE = (
    "I couldn't find a confident answer in our help content, so I've passed your "
    "question to our support team — someone will follow up with you shortly."
)


class QueryIn(BaseModel):
    question: str
    conversation_id: Optional[str] = None


@app.post("/query")
def query(body: QueryIn):
    question = body.question.strip()
    if not question:
        return {"escalated": False, "answer": "Please enter a question.", "citations": []}

    retrieval = hybrid_search(question)
    confidence = retrieval["max_vscore"]  # embedding similarity = the gate signal

    # Gate 1: best retrieval similarity too low -> escalate without an LLM call.
    if not retrieval["results"] or confidence < settings.min_vscore:
        esc_id = record_and_email(
            question, body.conversation_id, "low_retrieval_confidence", confidence
        )
        return {"escalated": True, "escalation_id": esc_id, "message": HANDOFF_MESSAGE}

    result = answer(question, retrieval["results"])

    # Gate 2: model could not answer from the documents (no citations / abstained).
    if not result["answered"]:
        esc_id = record_and_email(
            question, body.conversation_id, "model_could_not_answer", confidence
        )
        return {"escalated": True, "escalation_id": esc_id, "message": HANDOFF_MESSAGE}

    return {
        "escalated": False,
        "answer": result["answer"],
        "citations": result["citations"],
        "confidence": confidence,
    }


class ReindexIn(BaseModel):
    doc_id: str
    source_type: str  # faq | manual | support_ticket
    path: Optional[str] = None       # file path to (re)load, OR
    text: Optional[str] = None       # raw text inline
    title: Optional[str] = None
    updated_at: Optional[str] = None  # ISO 8601; defaults to now
    deleted: bool = False             # true => remove from index


@app.post("/webhook/reindex")
def reindex(body: ReindexIn):
    """Call this from your source system's update/delete webhook."""
    if body.deleted:
        remove_document(body.doc_id)
        return {"doc_id": body.doc_id, "removed": True}

    if body.text is not None:
        raw = body.text
    elif body.path:
        raw = load_file(body.path)
    else:
        return {"error": "provide either `text` or `path`"}

    updated_at = (
        datetime.fromisoformat(body.updated_at)
        if body.updated_at
        else datetime.now(timezone.utc)
    )
    return ingest_document(
        body.doc_id,
        body.source_type,
        raw,
        title=body.title,
        source_uri=body.path,
        updated_at=updated_at,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
