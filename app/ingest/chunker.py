"""Section-aware chunking.

Chunk boundaries follow the manual's own structure (heading, then table, then
paragraph) before falling back to a size window. Fixed character windows split
"do not exceed 2%" away from the ingredient it qualifies, which is precisely the
failure mode that produces a confidently wrong claim verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .loader import Block

# Cheap token proxy: ~4 chars/token for English marketing/technical prose. Good
# enough for sizing decisions and avoids a tokenizer dependency per provider.
CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
    chunk_id: str
    manual_id: str
    product: str
    source_file: str
    section: str | None
    page: int | None
    kind: str
    text: str

    def citation(self) -> str:
        parts = [self.product]
        if self.section:
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return " > ".join(parts)

    def as_dict(self) -> dict:
        return asdict(self)


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n")]
    paras = [p for p in paras if p]
    return paras or ([text.strip()] if text.strip() else [])


def _pack(paragraphs: list[str], max_tokens: int, overlap: float) -> list[str]:
    """Greedily pack paragraphs up to the budget, carrying an overlap tail into
    the next chunk so a claim spanning a boundary is still retrievable."""
    chunks: list[str] = []
    current: list[str] = []
    budget = 0
    for para in paragraphs:
        size = approx_tokens(para)
        if size > max_tokens:
            # A single oversized paragraph (dense table or wall of text): split
            # it on a character window rather than emitting an unusable chunk.
            if current:
                chunks.append("\n\n".join(current))
                current, budget = [], 0
            window = max_tokens * CHARS_PER_TOKEN
            step = max(1, int(window * (1 - overlap)))
            for start in range(0, len(para), step):
                piece = para[start:start + window].strip()
                if piece:
                    chunks.append(piece)
            continue
        if budget + size > max_tokens and current:
            chunks.append("\n\n".join(current))
            tail = current[-1] if overlap > 0 else None
            current = [tail] if tail and approx_tokens(tail) <= max_tokens * overlap * 2 else []
            budget = approx_tokens(current[0]) if current else 0
        current.append(para)
        budget += size
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


def chunk_blocks(blocks: Iterable[Block], *, manual_id: str, product: str,
                 source_file: str, max_tokens: int = 500, overlap: float = 0.15,
                 min_chars: int = 40) -> tuple[list[Chunk], int]:
    """Returns (chunks, n_dropped). Dropped fragments are counted rather than
    silently discarded so ingestion can report how much of a deck was noise."""
    chunks: list[Chunk] = []
    dropped = 0
    for block in blocks:
        if not block.text.strip():
            continue
        if block.kind == "table":
            # Tables are never merged with prose and never overlap-split: a row
            # only means anything alongside its header row.
            pieces = _pack([block.text], max_tokens * 2, 0.0)
        else:
            pieces = _pack(_split_paragraphs(block.text), max_tokens, overlap)
        for piece in pieces:
            # Heading is prepended to the embedded text so section context is part
            # of the vector, not just metadata sitting beside it.
            body = f"{block.heading}\n{piece}" if block.heading else piece
            # Slide decks are padded with "THE END", "Thank you" and title
            # fragments. They carry no evidence, and because BM25 normalises by
            # document length a tiny chunk matching one query term outranks the
            # chunk that actually settles the claim. Measured after the heading is
            # attached: a terse line under a real section ("Concentration / 5%
            # niacinamide") is evidence, a bare title fragment is not.
            if len(body.strip()) < min_chars:
                dropped += 1
                continue
            index = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{manual_id}:{index:04d}",
                manual_id=manual_id,
                product=product,
                source_file=source_file,
                section=block.heading,
                page=block.page,
                kind=block.kind,
                text=body,
            ))
    return chunks, dropped
