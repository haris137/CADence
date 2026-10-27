"""Excel / CSV parser: no hard-coded layout, odd part-number storage, merged cells, decoys, near matches."""
import io
import os

import openpyxl

from src import excel_parser as X


def _ops(facts):
    return [f.value for f in facts if f.category == "operation"]


def test_merged_part_number_cells_give_every_operation(samples_dir):
    r = X.parse_excel(os.path.join(samples_dir, "excel", "manufacturing.xlsx"), "2280")
    assert len(_ops(r.facts)) == 4                      # Routing!A2:A5 is one merged cell
    mat = [f for f in r.facts if f.category == "material"]
    assert mat and mat[0].value == "AL 6061-T6" and mat[0].location == 'sheet "Parts", row 5'


def test_blank_continuation_rows_belong_to_the_part_above(samples_dir):
    r = X.parse_excel(os.path.join(samples_dir, "excel", "manufacturing.xlsx"), "2281")
    assert len(_ops(r.facts)) == 4                      # part number only on the first routing row


def test_part_number_stored_as_text_and_as_float(samples_dir):
    path = os.path.join(samples_dir, "excel", "manufacturing.xlsx")
    assert len(_ops(X.parse_excel(path, "2282").facts)) == 4   # "2282" text cells
    assert len(_ops(X.parse_excel(path, "2283").facts)) == 4   # 2283.0 float cells
    assert len(_ops(X.parse_excel(path, 2283).facts)) == 4     # int argument works too


def test_messy_headers_whitespace_and_decoys(samples_dir):
    r = X.parse_excel(os.path.join(samples_dir, "excel", "manufacturing_messy.xlsx"), "2280")
    cats = {f.category for f in r.facts}
    assert {"material", "stock_size", "finished_size", "operation", "machine", "cycle_time"} <= cats
    # 'Lookup' has 2280 as a quantity: must not be used because a real part-number column exists
    assert not any('sheet "Lookup"' in f.location for f in r.facts)
    # '2280-A' is reported as a near match, never silently used
    assert any(n["value"] == "2280-A" for n in r.near_matches)
    assert all(f.confidence == "low" for f in r.facts if f.category == "near_match")


def test_near_match_can_be_analysed_explicitly(samples_dir):
    r = X.parse_excel(os.path.join(samples_dir, "excel", "manufacturing_messy.xlsx"), "2280-A")
    assert any("anodised variant" in f.value for f in r.facts)


def test_csv_flat_export_dedupes_part_level_columns(samples_dir):
    r = X.parse_excel(os.path.join(samples_dir, "excel", "manufacturing.csv"), "2280")
    assert len([f for f in r.facts if f.category == "material"]) == 1
    assert len(_ops(r.facts)) == 4


def test_form_style_sheet_label_value_pairs():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Manufacturing record"])
    ws.append(["Part No:", "7001"])
    ws.append(["Material:", "C45 steel"])
    ws.append(["Stock size", "Ø50 x 210"])
    ws.append([])
    ws.append(["Op", "Operation", "Machine"])
    ws.append([10, "Face and center", "Lathe"])
    ws.append([20, "Turn Ø45", "Lathe"])
    buf = io.BytesIO()
    wb.save(buf)
    r = X.parse_excel(buf.getvalue(), "7001", "record.xlsx")
    vals = {f.category: f.value for f in r.facts}
    assert vals.get("material") == "C45 steel"
    assert len(_ops(r.facts)) == 2


def test_file_name_fallback_when_part_number_is_not_in_any_cell():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Material", "AL 7075"])
    ws.append(["Finish", "Anodise"])
    buf = io.BytesIO()
    wb.save(buf)
    r = X.parse_excel(buf.getvalue(), "5150", "5150_record.xlsx")
    assert r.matched_by == "filename"
    assert any(f.value == "AL 7075" for f in r.facts)


def test_unreadable_file_is_reported_not_raised():
    r = X.parse_excel(b"\x00\x01garbage" * 50, "2280", "broken.xlsx")
    assert r.facts == [] and r.warnings and "could not read" in r.warnings[0]
