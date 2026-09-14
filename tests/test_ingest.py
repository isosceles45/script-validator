from __future__ import annotations

from pathlib import Path

from app.ingest.chunker import chunk_blocks
from app.ingest.loader import Block, load, looks_like_heading, product_name_from_filename
from app.ingest.pipeline import ingest
from tests.fakes import FakeEmbedder


def test_product_name_strips_date_brand_and_doctype_noise():
    cases = {
        "(202307)_TFS_Vitamin_Lip_Sleeping_Mask_(en).pptx": "Vitamin Lip Sleeping Mask",
        "[Training] TFS_Tea_Tree_Toner_Pads_202206.pptx": "Tea Tree Toner Pads",
        "2019-08_Herb_Day_365_Master_Blending_Mask_(3_SKU).pptx": "Herb Day 365 Blending Mask",
        "ENGTFS_Vita_Drop_Sunquid_2403.pptx": "Vita Drop Sunquid",
    }
    for filename, expected in cases.items():
        assert product_name_from_filename(Path(filename)) == expected


def test_product_name_keeps_interior_digits():
    # "365" and "7" are part of the product name, not date noise.
    assert "365" in product_name_from_filename(Path("2019-08_Herb_Day_365_Mask.pptx"))
    assert "7" in product_name_from_filename(Path("TFS_PDRN_Hyalu_7_serum_250818.pptx"))


def test_heading_detection_rejects_sentences():
    assert looks_like_heading("Usage Instructions")
    assert looks_like_heading("3.1 Safety Information")
    assert not looks_like_heading("Apply two drops to clean skin every morning.")
    assert not looks_like_heading("")


def test_chunker_keeps_section_and_page_provenance():
    blocks = [Block(text="Apply two drops.", page=4, heading="Usage Instructions")]
    chunks = chunk_blocks(blocks, manual_id="m1", product="Ampoule",
                          source_file="m1.pptx", max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].section == "Usage Instructions"
    assert chunks[0].page == 4
    assert chunks[0].citation() == "Ampoule > Usage Instructions > p.4"
    # The heading is embedded with the body, not just stored beside it.
    assert chunks[0].text.startswith("Usage Instructions")


def test_chunker_splits_oversized_blocks_and_never_drops_text():
    body = " ".join(f"word{i}" for i in range(4000))
    chunks = chunk_blocks([Block(text=body)], manual_id="m", product="P",
                          source_file="f", max_tokens=100, overlap=0.15)
    assert len(chunks) > 1
    assert "word0" in chunks[0].text
    assert "word3999" in chunks[-1].text


def test_tables_stay_separate_from_prose():
    blocks = [Block(text="Prose body.", heading="Overview"),
              Block(text="Ingredient | Amount\nNiacinamide | 5%", kind="table",
                    heading="Overview")]
    chunks = chunk_blocks(blocks, manual_id="m", product="P", source_file="f")
    kinds = [c.kind for c in chunks]
    assert "table" in kinds and "prose" in kinds
    table_chunk = next(c for c in chunks if c.kind == "table")
    assert "Prose body" not in table_chunk.text


def test_ingest_is_idempotent_by_checksum(settings, store, manuals_dir):
    embedder = FakeEmbedder()
    first = ingest(settings, embedder, store)
    assert len(first.ingested) == 2
    assert first.total_chunks > 0

    second = ingest(settings, embedder, store)
    assert second.ingested == []
    assert len(second.skipped_unchanged) == 2
    # Unchanged manuals must not be re-embedded -- that is the whole point.
    assert len(embedder.calls) == 2

    forced = ingest(settings, embedder, store, force=True)
    assert len(forced.ingested) == 2


def test_ingest_reports_files_needing_conversion(settings, store, manuals_dir):
    (manuals_dir / "2020-04_Mascara.ppt").write_bytes(b"\xd0\xcf\x11\xe0legacy")
    report = ingest(settings, FakeEmbedder(), store)
    assert "2020-04_Mascara.ppt" in report.needs_conversion
    assert report.failed == []


def test_ingest_flags_manuals_with_no_extractable_text(settings, store, manuals_dir):
    (manuals_dir / "Empty_Deck.txt").write_text("   \n\n  ", encoding="utf-8")
    report = ingest(settings, FakeEmbedder(), store)
    assert "Empty_Deck.txt" in report.needs_ocr


def test_reingestion_replaces_stale_chunks(settings, store, manuals_dir):
    embedder = FakeEmbedder()
    ingest(settings, embedder, store)
    before = store.stats()["n_chunks"]

    target = manuals_dir / "TFS_Tea_Tree_Pore_Ampoule_202207.txt"
    target.write_text("Product Overview\nShorter revised manual.\n", encoding="utf-8")
    ingest(settings, embedder, store)

    after = store.stats()["n_chunks"]
    assert after < before
    texts = " ".join(r["text"] for r in
                     store._conn.execute("SELECT text FROM chunks").fetchall())
    # A retired claim must not survive re-ingestion and stay "supported".
    assert "80% tea tree leaf water" not in texts
