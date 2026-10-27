"""Optional live test against the real Gemini API. Skipped unless GEMINI_API_KEY is set.

    set GEMINI_API_KEY=...        (Windows)   |   export GEMINI_API_KEY=...   (macOS/Linux)
    pytest tests/test_live_llm.py -s
"""
import os

import pytest

from src import pipeline as PL, qa
from src.llm import LLMClient, providers_from_env

pytestmark = pytest.mark.skipif(not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")),
                                reason="no GEMINI_API_KEY / GROQ_API_KEY set")


def test_live_material_and_not_found(std_items):
    llm = LLMClient(providers_from_env())
    pa = PL.analyze_part("2280", std_items, PL.Settings(), llm)
    a = qa.answer_question("What material is it?", pa.facts, "2280", pa.found, llm=llm)
    print("\nLIVE:", a.provider, "|", a.text, "| verified:", a.verified)
    assert "6061" in a.text and a.verified
    a2 = qa.answer_question("Who is the operator's supervisor?", pa.facts, "2280", pa.found, llm=llm)
    print("LIVE:", a2.text)
    assert qa.NOT_FOUND in a2.text or not a2.cited or a2.mode == "template"
