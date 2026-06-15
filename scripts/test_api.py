"""Exercise the FastAPI routes in-process (no server/port needed)."""
import json

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def show(label, resp):
    print("\n===== " + label + "  ->  HTTP " + str(resp.status_code) + " =====")
    print(json.dumps(resp.json(), indent=2))


show("GET /health", client.get("/health"))
show(
    "POST /query  (in scope)",
    client.post("/query", json={"question": "How long do refunds take?"}),
)
show(
    "POST /query  (out of scope -> escalate)",
    client.post(
        "/query",
        json={"question": "What is the CEO personal phone number?", "conversation_id": "demo-1"},
    ),
)
show(
    "POST /webhook/reindex  (unchanged doc -> idempotent skip)",
    client.post(
        "/webhook/reindex",
        json={"doc_id": "faqs/shipping.txt", "source_type": "faq", "path": "data/faqs/shipping.txt"},
    ),
)
