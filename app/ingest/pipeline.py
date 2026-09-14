"""Ingestion: manuals -> blocks -> chunks -> embeddings -> vector store.

Runs as a batch job, never in the request path. Idempotent by checksum, so
re-running after adding one deck re-embeds only that deck.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..providers import Embedder
from ..store.vector_store import VectorStore
from .chunker import Chunk, chunk_blocks
from .loader import NEEDS_CONVERSION, SUPPORTED, discover, file_checksum, load, product_name_from_filename

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    ingested: list[dict[str, Any]] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    needs_conversion: list[str] = field(default_factory=list)
    needs_ocr: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    total_chunks: int = 0
    duration_s: float = 0.0
    embed_model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ingested": self.ingested,
            "skipped_unchanged": self.skipped_unchanged,
            "needs_conversion": self.needs_conversion,
            "needs_ocr": self.needs_ocr,
            "failed": self.failed,
            "total_chunks": self.total_chunks,
            "duration_s": round(self.duration_s, 2),
            "embed_model": self.embed_model,
        }

    def summary(self) -> str:
        lines = [
            f"ingested {len(self.ingested)} manual(s), {self.total_chunks} chunks "
            f"in {self.duration_s:.1f}s using {self.embed_model}",
        ]
        if self.skipped_unchanged:
            lines.append(f"  unchanged (skipped): {len(self.skipped_unchanged)}")
        if self.needs_conversion:
            lines.append(
                f"  NEEDS CONVERSION ({len(self.needs_conversion)}): legacy binary "
                f"Office files, convert with LibreOffice then re-run --")
            lines += [f"    - {name}" for name in self.needs_conversion]
        if self.needs_ocr:
            lines.append(
                f"  NEEDS OCR ({len(self.needs_ocr)}): no extractable text, likely "
                f"image-only; these contribute nothing to retrieval --")
            lines += [f"    - {name}" for name in self.needs_ocr]
        if self.failed:
            lines.append(f"  FAILED ({len(self.failed)}):")
            lines += [f"    - {f['file']}: {f['error']}" for f in self.failed]
        return "\n".join(lines)


def manual_id_for(path: Path, manuals_dir: Path) -> str:
    """Stable across re-runs and unique per file, unlike the product label
    (two decks can legitimately describe the same product)."""
    try:
        rel = path.relative_to(manuals_dir)
    except ValueError:
        rel = Path(path.name)
    return str(rel).replace("/", "__")


def ingest(settings: Settings, embedder: Embedder, store: VectorStore, *,
           force: bool = False, only: list[str] | None = None) -> IngestReport:
    started = time.time()
    report = IngestReport(embed_model=embedder.name)

    for path in sorted(settings.manuals_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in NEEDS_CONVERSION:
            report.needs_conversion.append(path.name)

    files = discover(settings.manuals_dir)
    if only:
        wanted = {o.lower() for o in only}
        files = [f for f in files if any(w in f.name.lower() for w in wanted)]

    log.info("ingestion starting", extra={"n_files": len(files), "force": force,
                                          "embed_model": embedder.name})

    for path in files:
        manual_id = manual_id_for(path, settings.manuals_dir)
        try:
            checksum = file_checksum(path)
            if not force and store.existing_checksum(manual_id) == checksum:
                report.skipped_unchanged.append(path.name)
                log.info("manual unchanged, skipping", extra={"manual_id": manual_id})
                continue

            product = product_name_from_filename(path)
            blocks = load(path)
            chunks: list[Chunk] = chunk_blocks(
                blocks, manual_id=manual_id, product=product, source_file=path.name,
                max_tokens=settings.chunk_tokens, overlap=settings.chunk_overlap)

            if not chunks:
                # An image-only deck is a silent retrieval hole: the product looks
                # covered because the file exists, but no claim about it can ever
                # be supported. Surface it loudly instead.
                report.needs_ocr.append(path.name)
                log.warning("manual produced no chunks",
                            extra={"manual_id": manual_id, "file": path.name})
                continue

            t0 = time.time()
            embeddings = embedder.embed([c.text for c in chunks], stage="ingest_embed")
            store.upsert_manual(manual_id=manual_id, source_file=path.name,
                                checksum=checksum, product=product, chunks=chunks,
                                embeddings=embeddings, embed_model=embedder.name)

            report.ingested.append({"manual_id": manual_id, "product": product,
                                    "file": path.name, "n_blocks": len(blocks),
                                    "n_chunks": len(chunks),
                                    "embed_s": round(time.time() - t0, 2)})
            report.total_chunks += len(chunks)
            log.info("manual ingested",
                     extra={"manual_id": manual_id, "product": product,
                            "n_chunks": len(chunks)})
        except Exception as exc:
            log.exception("manual ingestion failed", extra={"file": path.name})
            report.failed.append({"file": path.name, "error": str(exc)[:300]})

    report.duration_s = time.time() - started
    log.info("ingestion complete", extra=report.as_dict())
    return report
