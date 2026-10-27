"""PDF drawing parser and validation of LLM-structured fields."""
import os

from src import pdf_parser as P


def test_title_block_notes_and_dimensions(samples_dir):
    r = P.parse_pdf(os.path.join(samples_dir, "drawings", "2280_mounting_plate.pdf"))
    f = {x.category: x for x in r.facts}
    assert f["part_number"].value == "2280"
    assert f["material"].value == "AL 6061-T6"
    assert f["revision"].value == "B"
    assert f["general_tolerance"].value == "±0.1"
    assert r.units == "mm"
    hole = f["hole_callout"]
    assert hole.meta["count"] == 4 and hole.meta["diameter"] == 8.5
    dims = sorted(x.num for x in r.facts if x.category == "linear_dim")
    assert {100.0, 60.0, 12.0} <= set(dims)
    assert "2280" not in [x.value for x in r.facts if x.category == "linear_dim"]  # title block number is not a dimension


def test_shaft_tolerance_is_not_a_hole(samples_dir):
    r = P.parse_pdf(os.path.join(samples_dir, "drawings", "2281_drive_shaft.pdf"))
    dia = [x for x in r.facts if x.value.startswith("Ø40")]
    assert dia and dia[0].category == "diameter"       # 'Ø40 h7' is a shaft fit, not a hole


def test_scanned_drawing_is_detected_and_rendered(samples_dir, monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "false")           # without OCR the page goes to the vision fallback
    r = P.parse_pdf(os.path.join(samples_dir, "extra", "2280_drawing_scanned.pdf"))
    assert r.needs_vision and r.page_images and r.facts == []


def test_scanned_drawing_is_read_by_ocr_when_tesseract_is_installed(samples_dir):
    import shutil
    import pytest
    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")
    r = P.parse_pdf(os.path.join(samples_dir, "extra", "2280_drawing_scanned.pdf"))
    assert r.ocr_used and not r.needs_vision and r.facts
    assert all(f.confidence == "low" for f in r.facts)               # OCR values are never shown as high confidence
    assert any(f.category == "part_number" and f.value.strip() == "2280" for f in r.facts)


def test_llm_structured_values_must_appear_in_the_drawing_text(samples_dir):
    r = P.parse_pdf(os.path.join(samples_dir, "drawings", "2280_mounting_plate.pdf"))
    fake_llm_json = {"material": "AL 6061-T6", "heat_treatment": "HRC 60",          # invented
                     "notes": ["DEBURR ALL HOLES", "PAINT RED"], "scale": "1:1"}       # second note invented
    new, dropped = P.structured_to_facts(fake_llm_json, r)
    texts = [f.value for f in new]
    assert "PAINT RED" not in texts and "HRC 60" not in texts
    assert any("HRC 60" in d for d in dropped) and any("PAINT RED" in d for d in dropped)


def test_not_a_pdf_does_not_crash():
    r = P.parse_pdf(b"this is not a pdf", "fake.pdf")
    assert r.facts == [] and r.warnings
