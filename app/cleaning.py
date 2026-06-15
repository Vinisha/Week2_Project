"""Ingestion cleaning: strip markup, decode entities, drop boilerplate.
Also loads raw text from PDFs and text files."""
import html
import re

from bs4 import BeautifulSoup

_PAGE_NUM_RE = re.compile(r"(page\s*)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    # Strip HTML/XML markup if present.
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(separator="\n")
    # Decode HTML entities (&amp; &#39; etc.).
    text = html.unescape(text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            cleaned_lines.append("")  # preserve paragraph breaks
            continue
        if _PAGE_NUM_RE.fullmatch(line):  # drop bare page-number boilerplate
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank runs
    text = re.sub(r"[ \t]{2,}", " ", text)   # collapse runs of spaces
    return text.strip()


def extract_text_from_pdf(path: str) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n\n".join(parts)


def load_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
