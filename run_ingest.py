"""Batch-ingest a data directory into the index.

Layout (subfolder name -> source_type):
    data/faqs/      -> faq
    data/manuals/   -> manual
    data/tickets/   -> support_ticket

Each file (.pdf, .txt, .md) becomes one document; doc_id is its relative path.
File mtime is used as `updated_at` so the staleness rule works out of the box.

Usage:
    python run_ingest.py data
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.cleaning import load_file
from app.db import init_db
from app.ingest import ingest_document

FOLDER_TO_TYPE = {
    "faqs": "faq",
    "manuals": "manual",
    "tickets": "support_ticket",
}
SUFFIXES = {".pdf", ".txt", ".md"}


def main(root: str) -> None:
    init_db()
    root_path = Path(root)
    total_files = total_chunks = 0

    for folder, source_type in FOLDER_TO_TYPE.items():
        directory = root_path / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in SUFFIXES:
                continue
            doc_id = str(path.relative_to(root_path))
            title = path.stem.replace("_", " ").replace("-", " ").title()
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            try:
                raw = load_file(str(path))
                result = ingest_document(
                    doc_id, source_type, raw, title=title,
                    source_uri=str(path), updated_at=updated_at,
                )
            except Exception as exc:
                print(f"  ERROR {doc_id}: {exc}")
                continue
            total_files += 1
            total_chunks += result.get("chunks", 0)
            note = result.get("skipped", f"{result.get('chunks', 0)} chunks")
            print(f"  {doc_id}: {note}")

    print(f"\nDone. {total_files} files processed, {total_chunks} chunks indexed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
