from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.store.run_store import RunStore  # noqa: E402
from app.store.vector_store import VectorStore  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(provider="openai", openai_api_key="test-key",
                    data_dir=tmp_path, db_path=tmp_path / "store.sqlite3",
                    top_k=3, golden_sample_size=2, min_similarity=0.25)


@pytest.fixture
def store(settings: Settings) -> VectorStore:
    vector_store = VectorStore(settings.db_path)
    yield vector_store
    vector_store.close()


@pytest.fixture
def run_store(settings: Settings) -> RunStore:
    (settings.data_dir / "runs").mkdir(parents=True, exist_ok=True)
    rs = RunStore(settings.db_path, settings.data_dir / "runs")
    yield rs
    rs.close()


@pytest.fixture
def manuals_dir(settings: Settings) -> Path:
    d = settings.manuals_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "TFS_Tea_Tree_Pore_Ampoule_202207.txt").write_text(
        "Product Overview\n"
        "The Tea Tree Pore Ampoule contains 80% tea tree leaf water sourced from "
        "Jeju Island. It helps reduce the appearance of enlarged pores.\n\n"
        "Usage Instructions\n"
        "Apply 2 to 3 drops to clean skin morning and evening. Avoid the eye area.\n\n"
        "Clinical Testing\n"
        "In a four week consumer study of 30 participants, 78 percent reported "
        "visibly smoother skin texture. Dermatologically tested for sensitive skin.\n",
        encoding="utf-8")
    (d / "TFS_Rice_Water_Bright_Cleansing_202211.txt").write_text(
        "Product Overview\n"
        "Rice Water Bright Cleansing Foam is formulated with rice extract and "
        "moringa oil to gently remove impurities.\n\n"
        "Usage Instructions\n"
        "Lather with water and massage onto damp skin, then rinse thoroughly.\n",
        encoding="utf-8")
    return d
