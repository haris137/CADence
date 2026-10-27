"""Cross-checker: unit normalisation first, then MATCH / MISMATCH / NOT COMPARABLE with tags."""
from src import crosscheck as C
from src.facts import Fact, assign_ids


def _status(pa, name_part):
    return [c for c in pa.checks if name_part in c.name]


def test_definition_of_done_level3_mismatch_has_all_three_tags(analysed):
    pa = analysed("2281")
    rows = _status(pa, "Largest overall dimension")
    assert rows and rows[0].status == "MISMATCH"
    tags = " ".join(rows[0].tags)
    assert "E" in tags and "P" in tags and "I" in tags
    assert "Excel disagrees" in rows[0].detail
    assert _status(pa, "Revision")[0].status == "MISMATCH"


def test_definition_of_done_inch_iges_gives_no_false_mismatch(analysed):
    pa = analysed("2283")
    assert not [c for c in pa.checks if c.status == "MISMATCH"]
    assert _status(pa, "Units")[0].status == "INFO"
    assert _status(pa, "Holes")[0].status == "MATCH"


def test_consistent_part_all_match(analysed):
    pa = analysed("2280")
    assert {c.status for c in pa.checks} == {"MATCH"}


def test_missing_pdf_is_a_missing_fact(analysed):
    pa = analysed("2282")
    assert pa.status["pdf"].state == "no_match"
    assert any(f.source_type == "missing" and "PDF drawing unavailable" in f.value for f in pa.facts)


def _f(src, cat, value, **meta):
    return Fact(source_type=src, source_file="t", location="t", field=cat, value=value, category=cat, meta=meta)


def test_material_notations_are_compared_by_alloy():
    facts = assign_ids([_f("excel", "material", "6061-T6 ALUMINIUM"), _f("pdf", "material", "AL 6061")])
    r = C.check_material(facts)[0]
    assert r.status == "MATCH"
    facts = assign_ids([_f("excel", "material", "AL 7075"), _f("pdf", "material", "AL 6061-T6")])
    assert C.check_material(facts)[0].status == "MISMATCH"


def test_stock_smaller_than_part_is_flagged_with_unit_hint():
    from src.units import parse_dims
    stock = _f("excel", "stock_size", "4 x 2.5 x 0.6", dims=dict(parse_dims("4 x 2.5 x 0.6"), unit="mm"))
    fin = _f("excel", "finished_size", "100 x 60 x 12", dims=dict(parse_dims("100 x 60 x 12"), unit="mm"))
    rows = C.check_dimensions(assign_ids([stock, fin]), C.Tolerance())
    stock_row = [r for r in rows if r.name.startswith("Stock")][0]
    assert stock_row.status == "MISMATCH" and "unit mix-up" in stock_row.detail
