"""Shared fixtures. Kept deliberately small so `make test` stays under a minute."""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# Tests use a separate in-memory index even when the student's .env uses Docker.
# Explicit env values also keep the default-backend tests independent of .env.
os.environ["QDRANT_MODE"] = "memory"
os.environ["EMBEDDING_BACKEND"] = "fastembed"


@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory) -> Path:
    """A 60-doc slice of the real corpus: same schema, ~1 s to index."""
    src = ROOT / "data" / "corpus_vn.jsonl"
    if not src.exists():
        pytest.skip("corpus missing — run `make seed`")
    docs = [json.loads(l) for l in src.open(encoding="utf-8")]
    # keep whole topic clusters so topic filters have something to match
    keep = [d for d in docs if int(d["doc_id"].split("_")[-1]) < 6]
    out = tmp_path_factory.mktemp("corpus") / "mini.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in keep:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return out


@pytest.fixture(scope="session")
def index(mini_corpus):
    from app.filters import FilteredIndex
    from app.search import Searcher
    return FilteredIndex.from_searcher(Searcher.from_corpus(mini_corpus))
