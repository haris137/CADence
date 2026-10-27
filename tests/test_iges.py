"""IGES parser — written from the IGES 5.3 structure, tested on hand-written records, generated samples and
real OpenCASCADE exports in tests/real_files/ (add your own FreeCAD / McMaster-Carr files there)."""
import glob
import os

import pytest

from src import iges_parser as I

HERE = os.path.dirname(__file__)


def rec(text: str, sec: str, seq: int) -> str:
    return text[:72].ljust(72) + sec + str(seq).rjust(7)


def prec(text: str, de: int, seq: int) -> str:
    return text[:64].ljust(64) + str(de).rjust(8) + "P" + str(seq).rjust(7)


def dline1(t, p, xf=0, status="00000000", seq=1):
    f = lambda v: str(v).rjust(8)
    return f(t) + f(p) + f(0) + f(0) + f(0) + f(0) + f(xf) + f(0) + status.rjust(8) + "D" + str(seq).rjust(7)


def dline2(t, count, form=0, seq=2):
    f = lambda v: str(v).rjust(8)
    return f(t) + f(0) + f(0) + f(count) + f(form) + " " * 24 + f(0) + "D" + str(seq).rjust(7)


def test_tokenizer_hollerith_d_exponent_and_defaults():
    toks = I.tokenize("110,1.5D0,-.25,3.,,5HA,B;C,1.0E-3;", ",", ";")
    assert toks == [110.0, 1.5, -0.25, 3.0, None, "A,B;C", 0.001]


def test_custom_delimiters_multiline_params_crlf_and_inch_units():
    # Global section with parameter delimiter '/' and record delimiter '!', units = inches
    g = "1H//1H!/9HBRKT-4410/9Htest.iges/4Htest/3H1.0/32/38/6/308/15/9HBRKT-4410/1.0/1/2HIN/1/0.01/" \
        "15H20261026.120000/1E-6/10./2Hme/3Horg/11/0/15H20261026.120000!"
    lines = [rec("custom delimiter test", "S", 1)]
    gl = [g[i:i + 72] for i in range(0, len(g), 72)]
    lines += [rec(x, "G", i + 1) for i, x in enumerate(gl)]
    # one line entity whose parameters span TWO P records
    lines += [dline1(110, 1, seq=1), dline2(110, 2, seq=2)]
    lines += [prec("110/0./0./0./", 1, 1), prec("4.0/2.0/0.5!", 1, 2)]
    lines += [rec(f"S{1:7d}G{len(gl):7d}D{2:7d}P{2:7d}", "T", 1)]
    data = ("\r\n".join(lines) + "\r\n").encode("latin-1")
    r = I.parse_iges(data, "custom.iges")
    assert r.global_params["parameter_delimiter"] == "/" and r.global_params["record_delimiter"] == "!"
    assert r.units == "in"
    assert [round(v, 3) for v in r.extents_mm] == [101.6, 50.8, 12.7]   # inches converted to mm
    assert "BRKT-4410" in r.part_numbers_seen


def test_transform_124_is_applied(samples_dir):
    r = I.parse_iges(os.path.join(samples_dir, "models", "2281_shaft.iges"))
    assert r.transforms_applied > 0
    assert [round(v, 3) for v in r.extents_mm] == [150.0, 40.0, 40.0]


def test_inch_model_is_normalised_to_mm(samples_dir):
    r = I.parse_iges(os.path.join(samples_dir, "models", "2283_bracket_inch.igs"))
    assert r.units == "in"
    assert [round(v, 3) for v in r.extents_mm] == [127.0, 50.8, 50.8]
    assert r.holes == [{"diameter_mm": 6.604, "count": 2, "circles": 2}] or r.holes[0]["count"] == 2


def test_crlf_and_d_exponents(samples_dir):
    r = I.parse_iges(os.path.join(samples_dir, "models", "2282_flange.IGS"))
    assert [round(v, 3) for v in r.extents_mm] == [120.0, 120.0, 20.0]
    holes = {h["diameter_mm"]: h["count"] for h in r.holes}
    assert holes[11.0] == 4 and holes[60.0] == 1


def test_malformed_line_and_entity_are_skipped_and_counted(samples_dir):
    with open(os.path.join(samples_dir, "models", "2280_plate.igs"), "rb") as fh:
        text = fh.read().decode("latin-1")
    lines = text.splitlines()
    lines.insert(5, "%%% garbage line that is not an IGES record %%%")
    # corrupt the parameters of the first line entity (P record 1)
    idx = next(i for i, l in enumerate(lines) if len(l) >= 73 and l[72] == "P")
    lines[idx] = "110,abc,xyz,0.,1.,2.,3.;".ljust(64) + lines[idx][64:]
    r = I.parse_iges("\n".join(lines).encode("latin-1"), "broken.igs")
    assert r.malformed_lines == 1
    assert sum(r.skipped.values()) >= 1
    assert r.extents_mm is not None                     # the rest of the file still parsed
    assert any(f.category == "skipped" for f in r.facts)


def test_volume_is_never_claimed(samples_dir):
    r = I.parse_iges(os.path.join(samples_dir, "models", "2280_plate.igs"))
    v = [f for f in r.facts if f.category == "volume_na"]
    assert v and "not reliably available" in v[0].value


def test_not_iges_does_not_crash():
    r = I.parse_iges(b"hello world\n" * 10, "x.igs")
    assert r.total_entities == 0 and r.warnings


REAL = sorted(glob.glob(os.path.join(HERE, "real_files", "*.igs")) + glob.glob(os.path.join(HERE, "real_files", "*.iges"))
              + glob.glob(os.path.join(HERE, "real_files", "*.IGS")) + glob.glob(os.path.join(HERE, "real_files", "*.IGES")))


@pytest.mark.parametrize("path", REAL, ids=[os.path.basename(p) for p in REAL] or None)
def test_real_iges_files(path, capsys):
    """Runs on every real file in tests/real_files/ and prints what was extracted."""
    r = I.parse_iges(path)
    with capsys.disabled():
        print(f"\n  {os.path.basename(path)}: units={r.units} ({r.unit_note}); entities={r.total_entities}; "
              f"type={r.model_type}")
        print(f"    counts={dict(sorted(r.entity_counts.items()))}")
        print(f"    bbox_mm={[round(v, 3) for v in (r.extents_mm or [])]} ({r.bbox_kind})")
        print(f"    holes={[(h['diameter_mm'], h['count']) for h in r.holes]}; skipped={r.skipped}; warnings={r.warnings[:3]}")
    assert r.total_entities > 0
    assert r.extents_mm is not None


def test_occ_exports_have_the_true_size():
    """Real OpenCASCADE exports (made with gmsh) of the sample parts: boxes and holes must be exact."""
    expect = {"occ_plate_100x60x12_4holes.igs": ([100, 60, 12], {8.5: 4}),
              "occ_shaft_150_dia40.iges": ([150, 40, 40], {30.0: 1, 40.0: 1}),
              "occ_flange_d120.igs": ([120, 120, 20], {11.0: 4, 60.0: 1, 120.0: 1}),
              "occ_filleted_block_80x50x20.igs": ([80, 50, 20], {})}
    for name, (box, holes) in expect.items():
        path = os.path.join(HERE, "real_files", name)
        if not os.path.exists(path):
            pytest.skip("OCC sample files not present")
        r = I.parse_iges(path)
        assert sorted([round(v, 3) for v in r.extents_mm], reverse=True) == box, name
        assert {h["diameter_mm"]: h["count"] for h in r.holes} == holes, name
