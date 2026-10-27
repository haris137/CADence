"""Shared fixtures: a freshly generated sample dataset and a fake OpenAI-compatible LLM server."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import make_samples  # noqa: E402
from src import pipeline as PL  # noqa: E402
from src.llm import LLMClient, Provider  # noqa: E402

STD = "Standard workbook (manufacturing.xlsx)"
MESSY = "Messy workbook (cryptic headers, decoys)"
CSV = "CSV export"


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("samples")
    make_samples.main(str(d))
    return str(d)


@pytest.fixture(scope="session")
def std_items(samples_dir):
    return PL.load_sample_set(samples_dir, STD)


@pytest.fixture(scope="session")
def messy_items(samples_dir):
    return PL.load_sample_set(samples_dir, MESSY)


@pytest.fixture(scope="session")
def csv_items(samples_dir):
    return PL.load_sample_set(samples_dir, CSV)


@pytest.fixture(scope="session")
def analysed(std_items):
    cache = {}

    def get(pn):
        if pn not in cache:
            cache[pn] = PL.analyze_part(pn, std_items, PL.Settings(use_llm=False))
        return cache[pn]
    return get


@pytest.fixture(scope="session")
def fake_llm_url():
    import fake_llm_server
    url, srv = fake_llm_server.start()
    yield url
    srv.shutdown()


@pytest.fixture
def make_llm(fake_llm_url):
    def make(*models):
        return LLMClient([Provider(m, fake_llm_url, "test-key", m) for m in models], timeout=5, backoff=0.05)
    return make
