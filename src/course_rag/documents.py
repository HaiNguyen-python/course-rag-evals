"""Load course documents and split them into overlapping chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def load_documents(corpus_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        text = read_text(path).strip()
        if not text:
            continue
        first = text.splitlines()[0].lstrip("# ").strip()
        docs.append(Document(path.stem, first or path.stem, text, content_hash(text)))
    return docs


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs; plain text becomes one section."""
    parts = re.split(r"^(#{1,6} .+)$", text, flags=re.MULTILINE)
    sections, heading = [], ""
    for part in parts:
        if re.match(r"^#{1,6} ", part):
            heading = part.lstrip("# ").strip()
        elif part.strip():
            sections.append((heading, part.strip()))
    return sections or [("", text)]


def chunk_document(doc: Document, words: int = 120, overlap: int = 30) -> list[Chunk]:
    """Chunks never cross a section heading, so a citation points to one topic."""
    if overlap >= words:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    for heading, body in _sections(doc.text):
        tokens = body.split()
        step = words - overlap
        for start in range(0, max(len(tokens) - overlap, 1), step):
            piece = " ".join(tokens[start : start + words])
            chunks.append(
                Chunk(f"{doc.doc_id}#{len(chunks)}", doc.doc_id, doc.title, heading, piece)
            )
    return chunks
