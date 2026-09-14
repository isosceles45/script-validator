"""Manual loading.

Emits a flat list of `Block`s carrying page + heading provenance, because claim
verification is only defensible if a verdict can cite a section and page number.
Tables are kept as their own blocks: spec sheets (dosage, concentrations,
dimensions) are exactly where numeric claims get adjudicated, and flattening them
into prose destroys the row/column association.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}
# LibreOffice-only legacy binary format; flagged at ingest rather than silently dropped.
NEEDS_CONVERSION = {".ppt", ".doc"}


@dataclass
class Block:
    text: str
    page: int | None = None
    heading: str | None = None
    kind: str = "prose"  # "prose" | "table"


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(manuals_dir: Path) -> list[Path]:
    return sorted(p for p in manuals_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED)


# A manual heading: short line, no trailing period, and either numbered
# ("3.1 Safety"), ALL CAPS, or Title Case.
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 &/,'()\-]{2,70}$")


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if s.endswith((".", ",", ";")):
        return False
    if s.endswith(":"):
        s = s[:-1].strip()
    if not _HEADING.match(s):
        return False
    words = s.split()
    if len(words) > 10:
        return False
    # Reject ordinary sentences that happen to start capitalised.
    lowered = sum(1 for w in words[1:] if w.islower() and w not in
                  {"and", "or", "of", "for", "the", "a", "in", "to", "with"})
    return lowered <= 1


def _table_to_text(table: list[list[str | None]]) -> str:
    """Render a table as pipe-delimited rows. Each row keeps its header context so
    a retrieved table chunk is self-describing."""
    rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in table if row]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    return "\n".join(" | ".join(r) for r in rows)


def load_pdf(path: Path) -> list[Block]:
    import pdfplumber

    blocks: list[Block] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                rendered = _table_to_text(table)
                if rendered:
                    blocks.append(Block(text=rendered, page=page_no, kind="table"))
            text = page.extract_text() or ""
            current_heading: str | None = None
            buffer: list[str] = []

            def flush() -> None:
                body = "\n".join(buffer).strip()
                if body:
                    blocks.append(Block(text=body, page=page_no, heading=current_heading))

            for line in text.splitlines():
                if looks_like_heading(line):
                    flush()
                    buffer = []
                    current_heading = line.strip().rstrip(":")
                else:
                    buffer.append(line)
            flush()
    if not any(b.text.strip() for b in blocks):
        log.warning("pdf produced no extractable text -- likely a scanned document "
                    "needing OCR", extra={"file": str(path)})
    return blocks


def load_docx(path: Path) -> list[Block]:
    import docx

    document = docx.Document(str(path))
    blocks: list[Block] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            blocks.append(Block(text=body, heading=current_heading))

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading") or style == "title" or looks_like_heading(text):
            flush()
            buffer = []
            current_heading = text.rstrip(":")
        else:
            buffer.append(text)
    flush()

    for table in document.tables:
        rendered = _table_to_text([[c.text for c in row.cells] for row in table.rows])
        if rendered:
            blocks.append(Block(text=rendered, heading=current_heading, kind="table"))
    return blocks


def load_pptx(path: Path) -> list[Block]:
    """PowerPoint training decks are the dominant manual format in this corpus.

    Each slide becomes its own provenance unit: the title placeholder is the
    section, the slide number is the page. Speaker notes are captured as their
    own block -- in training decks the on-slide text is often a marketing
    fragment while the substantiating detail (concentrations, test conditions,
    claim wording) sits in the notes.
    """
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation(str(path))
    blocks: list[Block] = []

    for slide_no, slide in enumerate(prs.slides, start=1):
        title: str | None = None
        if slide.shapes.title is not None:
            candidate = (slide.shapes.title.text or "").strip().replace("\n", " ")
            title = re.sub(r"\s+", " ", candidate)[:100] or None

        body: list[tuple[float, str]] = []

        def walk(shapes) -> None:
            for shape in shapes:
                if shape.shape_type == 6:  # GROUP -- flatten nested shapes
                    walk(shape.shapes)
                    continue
                if getattr(shape, "has_table", False) and shape.has_table:
                    rendered = _table_to_text(
                        [[c.text for c in row.cells] for row in shape.table.rows])
                    if rendered:
                        blocks.append(Block(text=rendered, page=slide_no,
                                            heading=title, kind="table"))
                    continue
                if not getattr(shape, "has_text_frame", False):
                    continue
                text = (shape.text_frame.text or "").strip()
                if not text or (title and text.strip() == title):
                    continue
                # Sort by vertical position so reading order survives the
                # arbitrary z-order that shapes are stored in.
                top = float(shape.top if shape.top is not None else Emu(0))
                body.append((top, text))

        walk(slide.shapes)

        body.sort(key=lambda pair: pair[0])
        merged = "\n\n".join(text for _, text in body).strip()
        if merged:
            blocks.append(Block(text=merged, page=slide_no, heading=title))

        if slide.has_notes_slide:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()
            if notes:
                blocks.append(Block(
                    text=f"[Speaker notes] {notes}", page=slide_no,
                    heading=title, kind="notes"))

    if not any(b.text.strip() for b in blocks):
        log.warning("pptx produced no extractable text -- deck may be image-only "
                    "and need OCR", extra={"file": str(path)})
    return blocks


def load_text(path: Path) -> list[Block]:
    blocks: list[Block] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            blocks.append(Block(text=body, heading=current_heading))

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or looks_like_heading(stripped):
            flush()
            buffer = []
            current_heading = stripped.lstrip("# ").rstrip(":")
        else:
            buffer.append(line)
    flush()
    return blocks


def load(path: Path) -> list[Block]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".pptx":
        return load_pptx(path)
    if suffix in NEEDS_CONVERSION:
        raise ValueError(
            f"{path.name}: legacy binary Office format. Convert it first, e.g. "
            f"`soffice --headless --convert-to pptx --outdir {path.parent} '{path}'`")
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"unsupported file type: {path}")


def product_name_from_filename(path: Path) -> str:
    """Derive a product label from the filename.

    Filenames are the only product signal available before the text is read, and
    this corpus prefixes most of them with date stamps and brand/doc-type tokens
    ("(202307)_TFS_Vitamin_Lip_Sleeping_Mask_(en)"). The label is used as a
    retrieval *hint* -- a soft filter that falls back to the full corpus -- never
    as ground truth for a claim, so imperfect derivation degrades ranking rather
    than correctness.
    """
    # Normalise separators up front so every rule below can assume whitespace.
    stem = re.sub(r"[_\-]+", " ", path.stem)
    # Mojibake from mis-encoded Korean filenames: drop non-ASCII rather than
    # surface "8êÿ8áò" as a product name.
    stem = "".join(ch if ch.isascii() else " " for ch in stem)
    stem = re.sub(r"[\[\]()]", " ", stem)

    rules = [
        r"(?<!\d)\d{4}\s\d{2}(?!\d)",                  # 2019-08 -> "2019 08"
        r"(?<!\d)\d{4,8}(?!\d)",                        # 202307, 240726, 2404
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        r"\bENG\s*TFS\b|\bENGTFS\b|\bTFS\b|\bfmgt\b",
        r"\bthe\s+face\s+shop\b",
        r"\b(manual|product|guide|datasheet|spec|sheet|training|info|data|intro|"
        r"introduction|updated|update|added|renewal|renewel|upload|eng|en|kor|"
        r"global|master|v\d+(\.\d+)*)\b",
        r"\b\d+\s*sku[s]?\b",
    ]
    for rule in rules:
        stem = re.sub(rule, " ", stem, flags=re.I)

    # Strip *trailing* orphan digits left behind by the rules above (mojibake
    # remnants, "(1)" duplicate markers), repeating until stable. Deliberately
    # trailing-only: interior digits are usually part of the name ("Herb Day
    # 365", "Hyalu 7") and stripping them would merge distinct products.
    while True:
        cleaned = re.sub(r"\s+\d+\s*$", " ", stem)
        if cleaned == stem:
            break
        stem = cleaned

    stem = re.sub(r"[,\s]+", " ", stem).strip(" ,")
    return stem.title() if stem else path.stem
