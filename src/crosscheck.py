"""Deterministic cross-checks between Excel, PDF and IGES facts (code only — no LLM).

Everything is converted to millimetres before comparing. Each check returns MATCH / MISMATCH /
NOT COMPARABLE (with a reason) or INFO, plus the fact tags that contributed.

Domain rule worth knowing: *stock size* is the raw block BEFORE machining, so it is normally larger
than the finished part. Stock is therefore checked as "stock >= finished", never as equality —
comparing stock to the CAD bounding box for equality would raise false mismatches on real data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .facts import Fact, missing_fact
from .units import UNIT_TO_MM, close, fmt_num, normalize_unit, parse_number

MATCH, MISMATCH, NOT_COMPARABLE, INFO = "MATCH", "MISMATCH", "NOT COMPARABLE", "INFO"
RANK_NAMES = ["Largest overall dimension (length)", "Middle overall dimension (width / Ø)",
              "Smallest overall dimension (thickness / Ø)"]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    tags: List[str] = field(default_factory=list)
    values: Dict[str, str] = field(default_factory=dict)

    def as_fact(self) -> Fact:
        return Fact(source_type="check", source_file="", location="cross-check", field=self.name,
                    value=f"{self.status} — {self.detail}", category="check", raw=self.detail,
                    meta={"status": self.status, "tags": self.tags, "values": self.values})


@dataclass
class Tolerance:
    rel: float = 0.01
    abs_mm: float = 0.5

    def ok(self, a: float, b: float) -> bool:
        return close(a, b, self.rel, self.abs_mm)

    def hole_ok(self, a: float, b: float) -> bool:
        return abs(a - b) <= max(0.05, self.rel * max(a, b))


def _tag(f: Fact) -> str:
    return f"[{f.id}]"


def _mm(value: float, unit: Optional[str]) -> float:
    return value * UNIT_TO_MM.get(unit or "mm", 1.0)


def _unit_label(unit: Optional[str], assumed: bool) -> str:
    return f"{unit or 'mm'}{' (assumed)' if assumed or not unit else ''}"


def _ratio_hint(a: float, b: float) -> str:
    if min(a, b) <= 0:
        return ""
    r = max(a, b) / min(a, b)
    for factor, txt in ((25.4, "inch/mm mix-up"), (10.0, "cm/mm mix-up"), (1000.0, "m/mm mix-up"), (2.0, "radius vs diameter")):
        if abs(r - factor) / factor < 0.01:
            return f" (ratio ≈ {fmt_num(r, 2)} → possible {txt})"
    return ""


# ------------------------------------------------------------------------------------------- collectors

def _excel_triplets(facts: List[Fact]) -> Tuple[Optional[Tuple[List[float], List[str], str]], Optional[Tuple[List[float], List[str], str]]]:
    """Return (finished, stock) as (sorted mm triplet, tags, note)."""
    finished = stock = None
    for f in facts:
        if f.source_type != "excel":
            continue
        dims = f.meta.get("dims")
        if not dims:
            continue
        unit = dims.get("unit")
        trip = sorted((_mm(v, unit) for v in dims["triplet"]), reverse=True)
        note = _unit_label(unit, dims.get("unit_assumed", unit is None))
        if f.category == "finished_size" and finished is None:
            finished = (trip, [_tag(f)], note)
        elif f.category == "stock_size" and stock is None:
            stock = (trip, [_tag(f)], note)
    if finished is None:
        # separate Length / Width / Thickness / Diameter columns in one row
        groups: Dict[str, Dict[str, Fact]] = {}
        for f in facts:
            if f.source_type == "excel" and f.category in ("length", "width", "thickness", "diameter") and f.num:
                groups.setdefault(f.meta.get("group", ""), {})[f.category] = f
        for g, d in groups.items():
            if "length" in d and ("width" in d or "diameter" in d):
                if "diameter" in d:
                    vals = [d["length"], d["diameter"], d["diameter"]]
                else:
                    vals = [d["length"], d["width"]] + ([d["thickness"]] if "thickness" in d else [])
                if len(vals) < 3:
                    continue
                trip = sorted((_mm(v.num, v.unit) for v in vals), reverse=True)
                finished = (trip, sorted({_tag(v) for v in vals}), _unit_label(vals[0].unit, vals[0].unit is None))
                break
    return finished, stock


def _pdf_overall(facts: List[Fact]) -> Tuple[Optional[Tuple[List[float], List[str], str]], Optional[Tuple[List[float], List[str], str]]]:
    overall = stock = None
    for f in facts:
        if f.source_type == "pdf" and f.category == "overall_size" and f.meta.get("dims"):
            dims = f.meta["dims"]
            unit = dims.get("unit")
            trip = sorted((_mm(v, unit) for v in dims["triplet"]), reverse=True)
            item = (trip, [_tag(f)], _unit_label(unit, dims.get("unit_assumed", unit is None)))
            if f.meta.get("stock"):
                stock = stock or item
            else:
                overall = overall or item
    return overall, stock


def _pdf_dimset(facts: List[Fact]) -> List[Tuple[float, Fact]]:
    out = []
    for f in facts:
        if f.source_type != "pdf" or f.num is None:
            continue
        if f.category in ("linear_dim", "toleranced_dim", "diameter") or (f.category == "hole_callout" and False):
            out.append((_mm(f.num, f.unit), f))
    return out


def _iges_box(facts: List[Fact]) -> Optional[Tuple[List[float], Dict[int, str], bool, Fact]]:
    box = next((f for f in facts if f.source_type == "iges" and f.category == "bbox" and f.meta.get("extents_mm")), None)
    if box is None:
        return None
    ext = list(box.meta["extents_mm"])
    order = sorted(range(3), key=lambda i: -ext[i])
    extent_facts = [f for f in facts if f.source_type == "iges" and f.category == "extent"]
    tags = {}
    for rank, axis in enumerate(order):
        name = "XYZ"[axis]
        ef = next((f for f in extent_facts if f.field.startswith(name)), None)
        tags[rank] = _tag(ef) if ef else _tag(box)
    return sorted(ext, reverse=True), tags, bool(box.meta.get("upper_bound")), box


# ------------------------------------------------------------------------------------------- checks

def _compare(vals: Dict[str, Tuple[float, str]], tol: Tolerance) -> Tuple[str, str]:
    """vals: source -> (mm value, tag). Returns (status, explanation)."""
    names = list(vals)
    if len(names) < 2:
        return NOT_COMPARABLE, "only one source states this value"
    agree = {a: [b for b in names if b != a and tol.ok(vals[a][0], vals[b][0])] for a in names}
    if all(len(v) == len(names) - 1 for v in agree.values()):
        return MATCH, "all sources agree within tolerance"
    if len(names) >= 3:
        odd = [a for a in names if not agree[a]]
        if len(odd) == 1:
            other = [b for b in names if b != odd[0]][0]
            return MISMATCH, f"{odd[0]} disagrees with the other sources" + _ratio_hint(vals[odd[0]][0], vals[other][0])
    a, b = names[0], names[1]
    return MISMATCH, "sources disagree" + _ratio_hint(vals[a][0], vals[b][0])


def check_dimensions(facts: List[Fact], tol: Tolerance) -> List[CheckResult]:
    out: List[CheckResult] = []
    excel_fin, excel_stock = _excel_triplets(facts)
    pdf_overall, pdf_stock = _pdf_overall(facts)
    pdf_dims = _pdf_dimset(facts)
    iges = _iges_box(facts)
    if not any([excel_fin, pdf_overall, pdf_dims, iges]):
        return [CheckResult("Overall dimensions", NOT_COMPARABLE, "no source gives overall dimensions")]
    sources_present = sum(bool(x) for x in (excel_fin, pdf_overall or pdf_dims, iges))
    if sources_present < 2:
        only = "Excel" if excel_fin else ("PDF" if (pdf_overall or pdf_dims) else "IGES")
        tags = (excel_fin[1] if excel_fin else []) + (pdf_overall[1] if pdf_overall else []) + ([iges[3].tag] if iges else [])
        out.append(CheckResult("Overall dimensions", NOT_COMPARABLE,
                               f"only {only} provides overall dimensions, nothing to compare against", tags))
    for rank in (range(3) if sources_present >= 2 else ()):
        vals: Dict[str, Tuple[float, str]] = {}
        notes: List[str] = []
        if excel_fin:
            vals["Excel"] = (excel_fin[0][rank], excel_fin[1][0] if len(excel_fin[1]) == 1 else " ".join(excel_fin[1]))
            if "assumed" in excel_fin[2]:
                notes.append("Excel units assumed mm")
        if iges:
            vals["IGES"] = (iges[0][rank], iges[1][rank])
        if pdf_overall:
            vals["PDF"] = (pdf_overall[0][rank], pdf_overall[1][0])
        elif pdf_dims:
            refs = [v[0] for k, v in vals.items()]
            hit = None
            for ref in refs:
                cand = [(abs(v - ref), v, f) for v, f in pdf_dims if tol.ok(v, ref)]
                if cand:
                    _, v, f = min(cand, key=lambda c: c[0])
                    hit = (v, _tag(f))
                    break
            if hit:
                vals["PDF"] = hit
            elif refs:
                notes.append("no matching dimension text on the drawing")
        status, why = _compare(vals, tol)
        if iges and iges[2] and status == MISMATCH and "IGES disagrees" in why and vals["IGES"][0] > max(v[0] for k, v in vals.items() if k != "IGES"):
            status, why = NOT_COMPARABLE, "IGES box is a control-point upper bound and comes out larger; not a reliable mismatch"
        shown = " vs ".join(f"{k} {fmt_num(v[0])} mm {v[1]}" for k, v in vals.items())
        if notes:
            why += "; " + "; ".join(notes)
        out.append(CheckResult(RANK_NAMES[rank], status, f"{shown} → {why}",
                               [x for (_v, tg) in vals.values() for x in tg.split()],
                               {k: fmt_num(v[0]) for k, v in vals.items()}))
    # stock vs finished
    stock = excel_stock or pdf_stock
    finished = None
    if iges:
        finished = (iges[0], [iges[1][r] for r in range(3)], "IGES")
    elif excel_fin:
        finished = (excel_fin[0], excel_fin[1], "Excel finished size")
    elif pdf_overall:
        finished = (pdf_overall[0], pdf_overall[1], "PDF overall size")
    if stock and finished:
        allowance = [stock[0][i] - finished[0][i] for i in range(3)]
        tags = stock[1] + finished[1]
        st = " × ".join(fmt_num(v) for v in stock[0])
        fi = " × ".join(fmt_num(v) for v in finished[0])
        if all(a >= -tol.abs_mm for a in allowance):
            big = [i for i in range(3) if stock[0][i] > 3 * max(finished[0][i], 1)]
            extra = "; allowance is unusually large on one axis" if big else ""
            out.append(CheckResult("Stock size vs finished part", MATCH,
                                   f"stock {st} mm {' '.join(stock[1])} is ≥ finished {fi} mm ({finished[2]}) — machining "
                                   f"allowance {', '.join(fmt_num(a) for a in allowance)} mm{extra}", tags))
        else:
            out.append(CheckResult("Stock size vs finished part", MISMATCH,
                                   f"stock {st} mm {' '.join(stock[1])} is SMALLER than finished {fi} mm ({finished[2]}) "
                                   f"on at least one axis — impossible; likely a wrong record or unit mix-up"
                                   + _ratio_hint(max(stock[0]), max(finished[0])), tags))
    elif stock and not finished:
        out.append(CheckResult("Stock size vs finished part", NOT_COMPARABLE,
                               "stock size is known but no finished-part size to compare it with", stock[1]))
    return out


_FAMILIES = [
    ("stainless", r"\bstainless\b|\bss\b|\binox\b|\b30[1-9]l?\b|\b31[0-9]l?\b|\b17-4\b"),
    ("aluminium", r"\balumin(i)?um\b|\bal\b|\balu\b|\b[1-7]\d{3}-?t\d+\b|\b6061\b|\b6082\b|\b7075\b|\b2024\b|\b5083\b"),
    ("steel", r"\bsteel\b|\bms\b|\baisi\b|\bsae\b|\ben\s?\d+|\bc45\b|\b10[0-9]{2}\b|\b41[0-9]{2}\b|\b43[0-9]{2}\b|\bmild\b"),
    ("brass", r"\bbrass\b|\bcuzn"),
    ("bronze", r"\bbronze\b"),
    ("copper", r"\bcopper\b|\bcu\b"),
    ("titanium", r"\btitanium\b|\bti-?6al|\bti\b"),
    ("cast iron", r"\bcast\s+iron\b|\bgg\d+|\bci\b"),
    ("plastic", r"\bdelrin\b|\bpom\b|\bnylon\b|\bpeek\b|\bptfe\b|\babs\b|\bacetal\b|\buhmw"),
]


def material_signature(text: str) -> Tuple[Optional[str], List[str], List[str]]:
    t = (text or "").lower()
    t = re.sub(r"^\s*(matl|mat'l|material)\s*[:=]\s*", "", t)
    family = None
    for name, rx in _FAMILIES:
        if re.search(rx, t):
            family = name
            break
    codes = re.findall(r"(?<![a-z0-9])(\d{3,4}[a-z]?)(?![0-9])", t)
    codes = [c for c in codes if not re.fullmatch(r"\d{3,4}", c) or 100 <= int(re.sub(r'\D', '', c)) <= 9999]
    tempers = re.findall(r"\b(t\d{1,4}|h\d{2,3}|o)\b", t)
    return family, codes, tempers


def check_material(facts: List[Fact]) -> List[CheckResult]:
    by_src: Dict[str, Fact] = {}
    for f in facts:
        if f.source_type in ("excel", "pdf") and f.category in ("material", "material_note") and f.source_type not in by_src:
            if f.category == "material_note":
                m = re.search(r"(?:MATL|MAT'L|MATERIAL)\s*[:=]?\s*(.+)", f.value, re.IGNORECASE)
                if not m:
                    continue
            by_src[f.source_type] = f
    has_iges = any(f.source_type == "iges" for f in facts)
    out: List[CheckResult] = []
    if len(by_src) < 2:
        if by_src:
            src, f = next(iter(by_src.items()))
            out.append(CheckResult("Material", NOT_COMPARABLE,
                                   f"only {'Excel' if src == 'excel' else 'PDF'} states the material ('{f.short_value(60)}' {f.tag}); "
                                   + ("IGES geometry files carry no material" if has_iges else "nothing to compare"), [f.tag]))
        return out
    e, p = by_src["excel"], by_src["pdf"]
    ev = re.sub(r"^\s*(matl|mat'l|material)\s*[:=]\s*", "", e.value, flags=re.I)
    pv = re.sub(r"^.*?(matl|mat'l|material)\s*[:=]\s*", "", p.value, flags=re.I) if p.category == "material_note" else p.value
    fe, ce, te = material_signature(ev)
    fp, cp, tp = material_signature(pv)
    shown = f"Excel '{ev.strip()}' {e.tag} vs PDF '{pv.strip()}' {p.tag}"
    if ev.strip().casefold() == pv.strip().casefold():
        status, why = MATCH, "identical text"
    elif ce and cp:
        if set(ce) & set(cp):
            status = MATCH
            why = "same alloy designation" + (" (temper/condition stated in only one source)" if set(te) != set(tp) else "")
            if fe and fp and fe != fp:
                status, why = MISMATCH, f"same number but different material family ({fe} vs {fp})"
        else:
            status, why = MISMATCH, f"different alloy designations ({', '.join(ce)} vs {', '.join(cp)})"
    elif fe and fp:
        status, why = (MATCH, f"same material family ({fe}); no alloy number to compare") if fe == fp else \
            (MISMATCH, f"different material families ({fe} vs {fp})")
    else:
        status, why = NOT_COMPARABLE, "material written in notations the checker cannot compare"
    note = "; IGES carries no material" if has_iges else ""
    out.append(CheckResult("Material", status, f"{shown} → {why}{note}", [e.tag, p.tag],
                           {"Excel": ev.strip(), "PDF": pv.strip()}))
    return out


def _excel_holes(facts: List[Fact]) -> List[Tuple[float, int, str]]:
    """Diameters mentioned in Excel text ('Drill 4x Ø8.5 THRU', 'bore Ø60 H7'), merged per diameter:
    (diameter, largest stated count, tags)."""
    merged: Dict[float, Tuple[int, List[str]]] = {}
    rx = re.compile(r"(?:(\d+)\s*[xX×]\s*)?(?:[Ø⌀∅]\s*|\bdia\.?\s*)(\d*\.?\d+)", re.IGNORECASE)
    for f in facts:
        if f.source_type != "excel" or f.category not in ("holes", "operation", "notes", "other"):
            continue
        for m in rx.finditer(f.value):
            d = parse_number(m.group(2))
            if d and d > 0:
                n = int(m.group(1)) if m.group(1) else 1
                cnt, tags = merged.get(d, (0, []))
                merged[d] = (max(cnt, n), tags + ([f.tag] if f.tag not in tags else []))
    return [(d, n, " ".join(tags)) for d, (n, tags) in sorted(merged.items())]


def check_holes(facts: List[Fact], tol: Tolerance, pdf_unit: Optional[str]) -> List[CheckResult]:
    pdf: Dict[float, Tuple[int, List[str]]] = {}
    for f in facts:
        if f.source_type == "pdf" and f.category == "hole_callout" and f.meta.get("diameter"):
            d = round(_mm(f.meta["diameter"], f.unit or pdf_unit), 3)
            key = next((k for k in pdf if tol.hole_ok(k, d)), d)
            cnt, tags = pdf.get(key, (0, []))
            pdf[key] = (cnt + int(f.meta.get("count", 1)), tags + [f.tag])
    iges: Dict[float, Tuple[int, str]] = {}
    for f in facts:
        if f.source_type == "iges" and f.category == "candidate_hole" and f.meta.get("diameter"):
            if f.meta.get("from") == "cylinder" and iges:
                continue
            iges[float(f.meta["diameter"])] = (int(f.meta.get("count", 0)), f.tag)
    excel = _excel_holes(facts)
    out: List[CheckResult] = []
    if not pdf and not iges and not excel:
        return out
    if not pdf and not excel:
        tags = [t for _, t in iges.values()]
        out.append(CheckResult("Holes", NOT_COMPARABLE, "only the IGES model shows circular features "
                               "(candidate holes); the drawing/record give no hole callouts to compare", tags))
        return out
    has_iges = any(f.source_type == "iges" for f in facts)
    for d, (cnt, tags) in sorted(pdf.items()):
        vals = [f"PDF Ø{fmt_num(d)} × {cnt} {' '.join(tags)}"]
        all_tags = list(tags)
        ig = next(((k, v) for k, v in iges.items() if tol.hole_ok(k, d)), None)
        ex = [e for e in excel if tol.hole_ok(_mm(e[0], pdf_unit), d)]
        status, why = NOT_COMPARABLE, "no IGES circular feature of this diameter" if has_iges else "IGES not available"
        if ig:
            vals.append(f"IGES Ø{fmt_num(ig[0])} × {ig[1][0]} {ig[1][1]}")
            all_tags.append(ig[1][1])
            status, why = (MATCH, "diameter and count agree") if ig[1][0] == cnt else \
                (MISMATCH, f"same diameter but count differs ({cnt} on drawing vs {ig[1][0]} in model)")
        elif has_iges and iges:
            status, why = MISMATCH, "drawing hole diameter not found among IGES circular features " + \
                f"({', '.join('Ø' + fmt_num(k) for k in iges)})"
        if ex:
            e = ex[0]
            vals.append(f"Excel Ø{fmt_num(e[0])} × {e[1]} {e[2]}")
            all_tags += e[2].split()
            if e[1] != cnt and e[1] != 1:
                status, why = MISMATCH, f"Excel count {e[1]} vs drawing {cnt}"
            elif status == NOT_COMPARABLE:
                status, why = MATCH, "Excel and drawing agree on the diameter"
        out.append(CheckResult(f"Holes Ø{fmt_num(d)} mm", status, " vs ".join(vals) + f" → {why}", all_tags))
    if not pdf and excel and iges:
        for d, n, tag in excel:
            dm = _mm(d, None)
            ig = next(((k, v) for k, v in iges.items() if tol.hole_ok(k, dm)), None)
            if ig:
                st = MATCH if ig[1][0] == n or n == 1 else MISMATCH
                why = "diameter found in both" + ("" if n == 1 else (" and the count agrees" if st == MATCH else
                                                                       f" but the count differs ({n} vs {ig[1][0]})"))
                out.append(CheckResult(f"Circular feature Ø{fmt_num(dm)} mm", st,
                                       f"Excel Ø{fmt_num(d)} × {n} {tag} vs IGES Ø{fmt_num(ig[0])} × {ig[1][0]} {ig[1][1]} → {why}",
                                       tag.split() + [ig[1][1]]))
            else:
                out.append(CheckResult(f"Circular feature Ø{fmt_num(dm)} mm", NOT_COMPARABLE,
                                       f"Excel mentions Ø{fmt_num(d)} {tag} but the IGES model has no circle of that "
                                       f"diameter (IGES: {', '.join('Ø' + fmt_num(k) for k in iges) or 'none'})", tag.split()))
    return out


def _norm_pn(s: str) -> str:
    return re.sub(r"[\s_]", "", str(s)).upper().removesuffix(".0")


def check_part_number(facts: List[Fact], part_number: str, iges_pns: Sequence[str], iges_tag: Optional[str],
                      pdf_pns: Sequence[str]) -> List[CheckResult]:
    target = _norm_pn(part_number)
    vals, tags, bad = {}, [], []
    e = next((f for f in facts if f.source_type == "excel" and f.category == "part_number"), None)
    if e:
        vals["Excel"] = e.value
        tags.append(e.tag)
        if _norm_pn(e.value) != target and not _norm_pn(e.value).endswith(target):
            bad.append("Excel")
    p = next((f for f in facts if f.source_type == "pdf" and f.category == "part_number"), None)
    if p:
        vals["PDF"] = p.value
        tags.append(p.tag)
        if target not in _norm_pn(p.value):
            bad.append("PDF")
    if iges_tag is not None:
        if any(target in _norm_pn(x) for x in iges_pns):
            vals["IGES"] = part_number
            tags.append(iges_tag)
    if len(vals) < 2 and not bad:
        return [CheckResult("Part number", NOT_COMPARABLE if len(vals) < 2 else MATCH,
                            "part number appears in fewer than two sources' content (files may have been matched by "
                            "file name)" + (f": {', '.join(f'{k} {v}' for k, v in vals.items())}" if vals else ""), tags)]
    shown = ", ".join(f"{k} '{v}'" for k, v in vals.items())
    if bad:
        return [CheckResult("Part number", MISMATCH, f"{shown} — {', '.join(bad)} shows a different part number than "
                            f"{part_number}; a wrong file may have been supplied", tags)]
    return [CheckResult("Part number", MATCH, f"{shown} — all name part {part_number}", tags)]


def check_revision(facts: List[Fact]) -> List[CheckResult]:
    e = next((f for f in facts if f.source_type == "excel" and f.category == "revision"), None)
    p = next((f for f in facts if f.source_type == "pdf" and f.category == "revision"), None)
    if not (e and p):
        return []
    ev, pv = e.value.strip().upper().lstrip("REV").strip(" .:"), p.value.strip().upper().lstrip("REV").strip(" .:")
    if ev == pv:
        return [CheckResult("Revision", MATCH, f"Excel rev {ev} {e.tag} = drawing rev {pv} {p.tag}", [e.tag, p.tag])]
    return [CheckResult("Revision", MISMATCH, f"Excel rev {ev} {e.tag} vs drawing rev {pv} {p.tag} — the record or "
                        f"the drawing may be outdated", [e.tag, p.tag])]


def check_units(facts: List[Fact], pdf_unit: Optional[str], iges_unit: Optional[str]) -> List[CheckResult]:
    parts, tags = [], []
    ex_units = sorted({f.unit for f in facts if f.source_type == "excel" and f.unit in UNIT_TO_MM})
    if any(f.source_type == "excel" for f in facts):
        parts.append(f"Excel: {', '.join(ex_units) if ex_units else 'not stated (mm assumed)'}")
    if any(f.source_type == "pdf" for f in facts):
        uf = next((f for f in facts if f.source_type == "pdf" and f.category == "units"), None)
        parts.append(f"PDF: {pdf_unit or 'not stated (mm assumed)'}" + (f" {uf.tag}" if uf else ""))
        if uf:
            tags.append(uf.tag)
    if any(f.source_type == "iges" for f in facts):
        uf = next((f for f in facts if f.source_type == "iges" and f.category == "units"), None)
        parts.append(f"IGES: {iges_unit or 'not stated (mm assumed)'}" + (f" {uf.tag}" if uf else ""))
        if uf:
            tags.append(uf.tag)
    known = {u for u in ex_units + [pdf_unit, iges_unit] if u}
    if not parts:
        return []
    if len(known) > 1:
        return [CheckResult("Units", INFO, "; ".join(parts) + " → different units; every value is converted to mm "
                            "in code before comparing, so this is not a mismatch", tags)]
    return [CheckResult("Units", MATCH if len(parts) > 1 else INFO, "; ".join(parts) +
                        " → compared in mm", tags)]


def run_checks(facts: List[Fact], part_number: str, tol: Tolerance, pdf_unit: Optional[str] = None,
               iges_unit: Optional[str] = None, iges_pns: Sequence[str] = (), pdf_pns: Sequence[str] = ()) -> List[CheckResult]:
    iges_units_fact = next((f for f in facts if f.source_type == "iges" and f.category == "units"), None)
    checks: List[CheckResult] = []
    for fn in (lambda: check_part_number(facts, part_number, iges_pns, iges_units_fact.tag if iges_units_fact else None, pdf_pns),
               lambda: check_units(facts, pdf_unit, iges_unit),
               lambda: check_dimensions(facts, tol),
               lambda: check_material(facts),
               lambda: check_holes(facts, tol, pdf_unit),
               lambda: check_revision(facts)):
        try:
            checks.extend(fn())
        except Exception as exc:  # a broken check must never break the app
            checks.append(CheckResult("Internal check", NOT_COMPARABLE, f"check failed: {exc}"))
    return checks


def missing_facts(facts: List[Fact], status: Dict[str, str], part_number: str) -> List[Fact]:
    """[MISSING] lines so the LLM (and the user) know what is absent instead of guessing."""
    out: List[Fact] = []
    names = {"excel": "Excel manufacturing record", "pdf": "PDF drawing", "iges": "IGES 3D model"}
    for src in ("excel", "pdf", "iges"):
        st = status.get(src, "missing")
        if st != "loaded":
            reason = {"missing": "no file of this type was provided",
                      "no_match": f"no file of this type matched part {part_number}",
                      "not_found": f"part {part_number} not found in it",
                      "error": "the file could not be read", "timeout": "reading the file timed out",
                      "empty": "the file contained no usable data"}.get(st, st)
            out.append(missing_fact(f"{names[src]} unavailable for part {part_number}: {reason}", src))
    have = {(f.source_type, f.category) for f in facts}
    if any(f.source_type == "pdf" for f in facts):
        if not any(c in ("general_tolerance", "toleranced_dim", "tolerance") for s, c in have if s == "pdf"):
            out.append(missing_fact("PDF: no tolerance information found", "pdf"))
        if not any(c in ("material", "material_note") for s, c in have if s == "pdf"):
            out.append(missing_fact("PDF: no material found on the drawing", "pdf"))
    if any(f.source_type == "excel" for f in facts):
        if not any(c in ("operation", "op_seq") for s, c in have if s == "excel"):
            out.append(missing_fact("Excel: no operation / routing rows found for this part", "excel"))
        if not any(c == "material" for s, c in have if s == "excel"):
            out.append(missing_fact("Excel: no material column/value found for this part", "excel"))
    if not any(c in ("price",) for _, c in have):
        out.append(missing_fact("No price, cost or supplier information exists in any provided file", "all sources"))
    return out
