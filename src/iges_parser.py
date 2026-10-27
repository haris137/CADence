"""Lightweight pure-Python IGES (5.x) reader — no CAD kernel needed.

Implements the parts of the IGES 5.3 file structure needed for reliable facts:
  * fixed 80-column records, section letter in column 73 (S, G, D, P, T), CRLF / trailing-whitespace tolerant
  * Global section with its own parameter / record delimiters and Hollerith strings (nHtext)
  * Directory entries (2 lines x 9 eight-column fields) incl. status flags, form number, transform pointer
  * Parameter data gathered from the P-section line range named by each Directory entry (multi-line)
  * free-form numbers such as 1.5D0, -.25, 3., 1.0E-3
  * entity 124 transformation matrices (chained), model-space scale and unit conversion to mm
  * bounding box from lines, arcs, copious data, points, B-rep vertices and B-spline control points
    (control points give a conservative upper bound and are labelled that way)
  * parameter-space curves (142 B-curves, use flag 05) and annotation/definition entities are excluded
  * candidate hole features from circles grouped by axis (a hole has two circular edges; split
    half-circles are merged), plus 5.3 right circular cylindrical surfaces (192)
Unsupported or malformed entities are skipped and counted; one bad entity never stops the file.
Volume, mass and surface area are deliberately NOT computed.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .facts import Fact
from .units import UNIT_TO_MM, fmt_num, iges_unit, parse_number

log = logging.getLogger(__name__)

ENTITY_NAMES = {
    0: "Null", 100: "Circular arc", 102: "Composite curve", 104: "Conic arc", 106: "Copious data",
    108: "Plane", 110: "Line", 112: "Parametric spline curve", 114: "Parametric spline surface",
    116: "Point", 118: "Ruled surface", 120: "Surface of revolution", 122: "Tabulated cylinder",
    123: "Direction", 124: "Transformation matrix", 125: "Flash", 126: "Rational B-spline curve",
    128: "Rational B-spline surface", 130: "Offset curve", 140: "Offset surface", 141: "Boundary",
    142: "Curve on parametric surface", 143: "Bounded surface", 144: "Trimmed surface",
    150: "Block", 152: "Right angular wedge", 154: "Right circular cylinder", 156: "Right circular cone frustum",
    158: "Sphere", 160: "Torus", 162: "Solid of revolution", 164: "Solid of linear extrusion", 168: "Ellipsoid",
    180: "Boolean tree", 182: "Selected component", 184: "Solid assembly",
    186: "Manifold solid B-rep object", 190: "Plane surface", 192: "Right circular cylindrical surface",
    194: "Right circular conical surface", 196: "Spherical surface", 198: "Toroidal surface",
    202: "Angular dimension", 204: "Curve dimension", 206: "Diameter dimension", 208: "Flag note",
    210: "General label", 212: "General note", 213: "New general note", 214: "Leader (arrow)",
    216: "Linear dimension", 218: "Ordinate dimension", 220: "Point dimension", 222: "Radius dimension",
    228: "General symbol", 230: "Sectioned area", 302: "Associativity definition", 304: "Line font definition",
    306: "Macro definition", 308: "Subfigure definition", 310: "Text font definition", 312: "Text display template",
    314: "Color definition", 316: "Units data", 320: "Network subfigure definition", 322: "Attribute table definition",
    402: "Associativity instance", 404: "Drawing", 406: "Property", 408: "Singular subfigure instance",
    410: "View", 412: "Rectangular array subfigure instance", 414: "Circular array subfigure instance",
    416: "External reference", 418: "Nodal load/constraint", 420: "Network subfigure instance",
    502: "Vertex list", 504: "Edge list", 508: "Loop", 510: "Face", 514: "Shell",
}
SOLID_TYPES = {186, 150, 152, 154, 156, 158, 160, 162, 164, 168, 180, 184}
BREP_TYPES = {186, 502, 504, 508, 510, 514}
SURFACE_TYPES = {108, 114, 118, 120, 122, 128, 140, 143, 144, 190, 192, 194, 196, 198}
CURVE_TYPES = {100, 102, 104, 106, 110, 112, 126, 130, 142}
ANNOTATION_TYPES = {106, 202, 204, 206, 208, 210, 212, 213, 214, 216, 218, 220, 222, 228, 230}
HANDLED_GEOMETRY = {100, 104, 106, 110, 116, 126, 128, 192, 502}
STRUCTURAL_OK = {0, 102, 108, 118, 120, 122, 123, 124, 141, 142, 143, 144, 186, 190, 194, 196, 198, 302, 304,
                 308, 310, 312, 314, 402, 404, 406, 410, 504, 508, 510, 514} | ANNOTATION_TYPES

Vec = Tuple[float, float, float]


@dataclass
class DirEntry:
    seq: int
    type: int
    param_ptr: int
    structure: int
    line_font: int
    level: int
    view: int
    transform: int
    label_assoc: int
    status: str
    line_weight: int
    color: int
    param_count: int
    form: int
    label: str
    subscript: int

    @property
    def blanked(self) -> bool:
        return self.status[0:2] == "01"

    @property
    def subordinate(self) -> str:
        return self.status[2:4]

    @property
    def use(self) -> str:
        return self.status[4:6]


@dataclass
class IgesResult:
    filename: str = ""
    facts: List[Fact] = field(default_factory=list)
    units: Optional[str] = None
    unit_note: str = ""
    global_params: Dict[str, Any] = field(default_factory=dict)
    entity_counts: Dict[int, int] = field(default_factory=dict)
    form_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    total_entities: int = 0
    bbox_min: Optional[Vec] = None
    bbox_max: Optional[Vec] = None
    bbox_kind: str = ""
    surface_bbox: Optional[Tuple[Vec, Vec]] = None
    extents_mm: Optional[List[float]] = None
    holes: List[Dict[str, Any]] = field(default_factory=list)
    cylinders: List[Dict[str, Any]] = field(default_factory=list)
    model_type: str = ""
    skipped: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    transforms_applied: int = 0
    transforms_failed: int = 0
    preview: List[List[Vec]] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    start_text: str = ""
    malformed_lines: int = 0
    part_numbers_seen: List[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------- low level

_HOLL = re.compile(r"(\d+)H", re.IGNORECASE)


def tokenize(s: str, pdelim: str = ",", rdelim: str = ";") -> List[Optional[Union[str, float]]]:
    """Split IGES free-format parameter text into values.

    Hollerith strings -> str, numbers -> float, empty (defaulted) -> None, other text -> str.
    Stops at the record delimiter (outside Hollerith strings).
    """
    out: List[Optional[Union[str, float]]] = []
    i, n = 0, len(s)
    while i <= n:
        while i < n and s[i] in " \t":
            i += 1
        if i >= n:
            break
        m = _HOLL.match(s, i)
        if m:
            length = int(m.group(1))
            start = m.end()
            out.append(s[start:start + length])
            i = start + length
            while i < n and s[i] in " \t":
                i += 1
            if i < n and s[i] == pdelim:
                i += 1
                continue
            break  # record delimiter or end
        k = i
        while k < n and s[k] != pdelim and s[k] != rdelim:
            k += 1
        tok = s[i:k].strip()
        if tok == "":
            out.append(None)
        else:
            v = parse_number(tok)
            out.append(v if v is not None else tok)
        if k >= n or s[k] == rdelim:
            break
        i = k + 1
        if i == n:  # trailing delimiter -> one more defaulted value
            out.append(None)
    return out


def _int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _f(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    p = parse_number(v)
    if p is None:
        raise ValueError(f"not a number: {v!r}")
    return p


def _split_records(text: str) -> Tuple[Dict[str, List[Tuple[int, str]]], int]:
    sections: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    malformed = 0
    for raw in text.splitlines():
        line = raw.rstrip("\r\n\x1a")
        if not line.strip():
            continue
        sec = None
        seq = None
        if len(line) >= 73 and line[72] in "SGDPTCB":
            sec = line[72]
            seq_txt = line[73:80].strip()
            seq = int(seq_txt) if seq_txt.isdigit() else None
        if sec is None or seq is None:
            m = re.search(r"([SGDPTC])\s{0,7}(\d{1,7})\s*$", line)
            if m and len(line) >= 60:
                sec, seq = m.group(1), int(m.group(2))
                body = line[: m.start()]
                line = body.ljust(72)[:72] + sec + str(seq).rjust(7)
            else:
                malformed += 1
                continue
        sections[sec].append((seq, line.ljust(80)))
    return sections, malformed


def _parse_global(gtext: str) -> Tuple[List[Any], str, str]:
    s = gtext.rstrip()
    pd, rd = ",", ";"
    i = 0
    if s.upper().startswith("1H") and len(s) >= 3:
        pd = s[2]
        i = 3
    # after field 1 comes pdelim (if field 1 present) then field 2
    if i < len(s) and s[i] == pd:
        i += 1
    elif i == 0 and s.startswith(","):
        i = 1
    rest = s[i:]
    if rest.upper().startswith("1H") and len(rest) >= 3:
        rd = rest[2]
    tokens = tokenize(s, pd, rd)
    return tokens, pd, rd


# ------------------------------------------------------------------------------------------- geometry helpers

Matrix = Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]


def _apply(m: Matrix, p: Vec) -> Vec:
    return (m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3])


def _apply_dir(m: Matrix, d: Vec) -> Vec:
    return (m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
            m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
            m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2])


def _norm3(v: Vec) -> Vec:
    l = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (0.0, 0.0, 1.0) if l < 1e-12 else (v[0] / l, v[1] / l, v[2] / l)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _canon_axis(n: Vec) -> Vec:
    n = _norm3(n)
    for c in n:
        if abs(c) > 1e-6:
            return n if c > 0 else (-n[0], -n[1], -n[2])
    return n


def _chunks(seq: Sequence[float], size: int) -> Iterable[Sequence[float]]:
    for i in range(0, len(seq) - size + 1, size):
        yield seq[i:i + size]


# ------------------------------------------------------------------------------------------- parser

def parse_iges(data_or_path: Union[bytes, str, os.PathLike], filename: Optional[str] = None,
               max_preview_points: int = 60000) -> IgesResult:
    if isinstance(data_or_path, (bytes, bytearray)):
        data, filename = bytes(data_or_path), filename or "model.igs"
    else:
        path = os.fspath(data_or_path)
        with open(path, "rb") as fh:
            data = fh.read()
        filename = filename or os.path.basename(path)
    res = IgesResult(filename=filename)
    text = data.decode("latin-1", errors="replace")
    sections, res.malformed_lines = _split_records(text)
    if res.malformed_lines:
        res.warnings.append(f"{res.malformed_lines} malformed line(s) ignored")
    if "C" in sections:
        res.warnings.append("compressed IGES format is not supported")
    if "B" in sections or (not sections.get("D") and not sections.get("G")):
        res.warnings.append("not a fixed-format ASCII IGES file (binary/compressed/other) — nothing extracted")
        return res

    # ---------------- Start + Global
    res.start_text = " ".join(l[:72].strip() for _, l in sorted(sections.get("S", [])))[:600]
    gtext = "".join(l[:72] for _, l in sorted(sections.get("G", [])))
    g, pd, rd = _parse_global(gtext)

    def gp(i: int) -> Any:
        return g[i - 1] if len(g) >= i else None

    res.global_params = {
        "parameter_delimiter": pd, "record_delimiter": rd,
        "product_id_sender": gp(3), "file_name": gp(4), "native_system": gp(5), "preprocessor": gp(6),
        "model_scale": gp(13), "units_flag": gp(14), "units_name": gp(15), "file_date": gp(18),
        "min_resolution": gp(19), "max_coordinate": gp(20), "author": gp(21), "organization": gp(22),
        "iges_version_flag": gp(23), "drafting_standard": gp(24), "model_date": gp(25),
        "product_id_receiver": gp(12),
    }
    flag = _int(gp(14), 0) if gp(14) is not None else None
    uname = gp(15) if isinstance(gp(15), str) else None
    res.units, res.unit_note = iges_unit(flag, uname)
    scale = parse_number(gp(13)) if gp(13) is not None else 1.0
    if not scale or scale <= 0:
        scale = 1.0
    unit_factor = UNIT_TO_MM.get(res.units, 1.0) if res.units else 1.0
    if res.units is None:
        res.warnings.append("IGES units not specified — coordinates treated as millimetres (assumed)")
    to_mm = unit_factor / scale

    # ---------------- Directory
    dlines = sorted(sections.get("D", []))
    des: Dict[int, DirEntry] = {}
    for i in range(0, len(dlines) - 1, 2):
        try:
            s1, l1 = dlines[i]
            s2, l2 = dlines[i + 1]
            if s2 != s1 + 1:
                res.skipped["directory pairing error"] = res.skipped.get("directory pairing error", 0) + 1
                continue

            def fld(line: str, k: int) -> str:
                return line[k * 8:(k + 1) * 8]

            status = fld(l1, 8).strip().replace(" ", "0").rjust(8, "0")
            de = DirEntry(seq=s1, type=_int(fld(l1, 0).strip()), param_ptr=_int(fld(l1, 1).strip()),
                          structure=_int(fld(l1, 2).strip()), line_font=_int(fld(l1, 3).strip()),
                          level=_int(fld(l1, 4).strip()), view=_int(fld(l1, 5).strip()),
                          transform=_int(fld(l1, 6).strip()), label_assoc=_int(fld(l1, 7).strip()),
                          status=status, line_weight=_int(fld(l2, 1).strip()), color=_int(fld(l2, 2).strip()),
                          param_count=_int(fld(l2, 3).strip(), 1), form=_int(fld(l2, 4).strip()),
                          label=fld(l2, 7).strip(), subscript=_int(fld(l2, 8).strip()))
            des[de.seq] = de
        except Exception as exc:  # noqa: BLE001
            res.skipped["unreadable directory entry"] = res.skipped.get("unreadable directory entry", 0) + 1
            log.debug("bad DE: %s", exc)
    if len(dlines) % 2:
        res.warnings.append("odd number of directory lines — last entry ignored")

    res.total_entities = len(des)
    cnt = Counter(d.type for d in des.values())
    res.entity_counts = dict(cnt)
    res.form_counts = dict(Counter((d.type, d.form) for d in des.values()))

    # ---------------- Parameter data
    plines = dict(sections.get("P", []))
    by_backptr: Dict[int, List[str]] = defaultdict(list)
    for seq in sorted(plines):
        bp = plines[seq][64:72].strip()
        if bp.isdigit():
            by_backptr[int(bp)].append(plines[seq])

    params: Dict[int, List[Any]] = {}

    def get_params(de: DirEntry) -> List[Any]:
        if de.seq in params:
            return params[de.seq]
        rows = [plines[k] for k in range(de.param_ptr, de.param_ptr + max(de.param_count, 1)) if k in plines]
        if not rows or (rows and rows[0][64:72].strip().isdigit() and int(rows[0][64:72].strip()) != de.seq
                        and by_backptr.get(de.seq)):
            rows = by_backptr.get(de.seq, rows)
        txt = "".join(r[:64] for r in rows)
        p = tokenize(txt, pd, rd)
        if p and _int(p[0], -1) != de.type:
            res.warnings.append(f"entity DE {de.seq}: parameter type {p[0]} != directory type {de.type}") if len(res.warnings) < 30 else None
        params[de.seq] = p
        return p

    # ---------------- transforms (124), chained through their own DE field 7
    matrices: Dict[int, Optional[Matrix]] = {}

    def matrix_of(seq: int, depth: int = 0) -> Optional[Matrix]:
        if seq in matrices:
            return matrices[seq]
        de = des.get(seq)
        if de is None or de.type != 124 or depth > 10:
            matrices[seq] = None
            return None
        try:
            p = get_params(de)
            v = [_f(x) for x in p[1:13]]
            if len(v) < 12:
                raise ValueError("short matrix")
            m: Matrix = ((v[0], v[1], v[2], v[3]), (v[4], v[5], v[6], v[7]), (v[8], v[9], v[10], v[11]))
            if de.transform:
                parent = matrix_of(de.transform, depth + 1)
                if parent is None:
                    raise ValueError("bad parent transform")
                m = _compose(parent, m)
            matrices[seq] = m
            return m
        except Exception:  # noqa: BLE001
            matrices[seq] = None
            return None

    def xf_point(de: DirEntry, p: Vec) -> Optional[Vec]:
        if not de.transform:
            return p
        m = matrix_of(de.transform)
        if m is None:
            return None
        return _apply(m, p)

    def xf_dir(de: DirEntry, d: Vec) -> Optional[Vec]:
        if not de.transform:
            return d
        m = matrix_of(de.transform)
        return None if m is None else _apply_dir(m, d)

    # ---------------- figure out which entities are parameter-space / definitions to exclude
    excluded: set = set()
    param_space: set = set()
    under_trim: set = set()
    for de in des.values():
        try:
            if de.type == 142:
                p = get_params(de)
                bptr = _int(p[3]) if len(p) > 3 else 0
                if bptr:
                    excluded.add(bptr)
                    param_space.add(bptr)
            elif de.type in (144, 143):
                p = get_params(de)
                sp = _int(p[1]) if len(p) > 1 else 0
                if sp:
                    under_trim.add(sp)
            elif de.type == 120:
                # surface of revolution: axis line + generatrix define an UNTRIMMED surface; they can extend
                # beyond the real faces (e.g. a hole tool longer than the plate), so they are not edges
                p = get_params(de)
                for ptr in p[1:3]:
                    if _int(ptr):
                        excluded.add(_int(ptr))
            elif de.type == 122:
                p = get_params(de)
                if len(p) > 1 and _int(p[1]):
                    excluded.add(_int(p[1]))
            elif de.type == 118:
                p = get_params(de)
                for ptr in p[1:3]:
                    if _int(ptr):
                        excluded.add(_int(ptr))
        except Exception:  # noqa: BLE001
            pass
    # propagate exclusion into composite curves (102) children
    stack = list(excluded)
    while stack:
        s = stack.pop()
        de = des.get(s)
        if de is None or de.type != 102:
            continue
        try:
            p = get_params(de)
            n = _int(p[1])
            for ptr in p[2:2 + n]:
                c = _int(ptr)
                if c and c not in excluded:
                    excluded.add(c)
                    if s in param_space:
                        param_space.add(c)
                    stack.append(c)
        except Exception:  # noqa: BLE001
            pass

    def geometric(de: DirEntry) -> bool:
        return de.use in ("00", "") and de.seq not in excluded

    # ---------------- extract geometry
    curve_pts: List[Vec] = []
    spline_used = False
    surf_pts: List[Vec] = []
    circles: List[Dict[str, Any]] = []
    preview: List[List[Vec]] = []
    preview_count = 0
    point_cache: Dict[int, Vec] = {}
    dir_cache: Dict[int, Vec] = {}

    def skip(reason: str) -> None:
        res.skipped[reason] = res.skipped.get(reason, 0) + 1

    def add_preview(poly: List[Vec]) -> None:
        nonlocal preview_count
        if preview_count + len(poly) <= max_preview_points:
            preview.append(poly)
            preview_count += len(poly)

    def transformed(de: DirEntry, pts: List[Vec]) -> Optional[List[Vec]]:
        out = []
        for p in pts:
            q = xf_point(de, p)
            if q is None:
                return None
            out.append((q[0] * to_mm, q[1] * to_mm, q[2] * to_mm))
        if de.transform:
            res.transforms_applied += 1
        return out

    for de in sorted(des.values(), key=lambda d: d.seq):
        t = de.type
        try:
            if t in (0,):
                continue
            if t not in HANDLED_GEOMETRY:
                if t not in STRUCTURAL_OK:
                    skip(f"unsupported entity {t} ({ENTITY_NAMES.get(t, 'unknown')})")
                continue
            if de.use == "05" or de.seq in param_space:
                skip("parameter-space curve (not model geometry)")
                continue
            if de.seq in excluded:
                skip("surface-definition curve (not an edge)")
                continue
            if de.use in ("01", "02", "06") and t != 502:
                skip({"01": "annotation entity", "02": "definition entity (subfigure)", "06": "construction geometry"}[de.use])
                continue
            p = get_params(de)
            if t == 110:
                if de.form in (1, 2):
                    skip("unbounded line (ray / infinite line)")
                    continue
                v = [_f(x) for x in p[1:7]]
                pts = transformed(de, [(v[0], v[1], v[2]), (v[3], v[4], v[5])])
                if pts is None:
                    res.transforms_failed += 1
                    continue
                curve_pts.extend(pts)
                add_preview(pts)
            elif t == 100:
                zt, x1, y1, x2, y2, x3, y3 = [_f(x) for x in p[1:8]]
                r = math.hypot(x2 - x1, y2 - y1)
                if r <= 0:
                    skip("degenerate arc")
                    continue
                a1 = math.atan2(y2 - y1, x2 - x1)
                a2 = math.atan2(y3 - y1, x3 - x1)
                full = math.hypot(x3 - x2, y3 - y2) <= 1e-9 * max(1.0, r)
                sweep = 2 * math.pi if full else (a2 - a1) % (2 * math.pi)
                if sweep <= 1e-12:
                    sweep = 2 * math.pi
                n = max(8, int(72 * sweep / (2 * math.pi)))
                local = [(x1 + r * math.cos(a1 + sweep * k / n), y1 + r * math.sin(a1 + sweep * k / n), zt) for k in range(n + 1)]
                pts = transformed(de, local)
                if pts is None:
                    res.transforms_failed += 1
                    continue
                curve_pts.extend(pts)
                add_preview(pts)
                c = transformed(de, [(x1, y1, zt)])
                s = transformed(de, [(x2, y2, zt)])
                nrm = xf_dir(de, (0.0, 0.0, 1.0))
                if de.transform:
                    res.transforms_applied -= 2  # counted once above
                if c and s and nrm:
                    rad = math.dist(c[0], s[0])
                    circles.append({"center": c[0], "normal": _norm3(nrm), "radius": rad,
                                    "sweep_deg": math.degrees(sweep), "de": de.seq})
            elif t == 104:
                A, B, C, D, E, F, zt, x1, y1, x2, y2 = [_f(x) for x in p[1:12]]
                local = [(x1, y1, zt), (x2, y2, zt)]
                if (de.form == 1 and abs(B) < 1e-12 and abs(D) < 1e-12 and abs(E) < 1e-12
                        and A * F < 0 and C * F < 0):
                    # ellipse in definition space: x²/a² + y²/b² = 1, traversed counter-clockwise
                    a, b = math.sqrt(-F / A), math.sqrt(-F / C)
                    t1, t2 = math.atan2(y1 / b, x1 / a), math.atan2(y2 / b, x2 / a)
                    sweep = (t2 - t1) % (2 * math.pi) or 2 * math.pi
                    n = max(8, int(72 * sweep / (2 * math.pi)))
                    local = [(a * math.cos(t1 + sweep * k / n), b * math.sin(t1 + sweep * k / n), zt) for k in range(n + 1)]
                else:
                    skip("conic arc (only end points used)")
                pts = transformed(de, local)
                if pts:
                    curve_pts.extend(pts)
                    add_preview(pts)
            elif t == 106:
                ip = _int(p[1])
                n = _int(p[2])
                raw: List[Vec] = []
                if ip == 1:
                    zt = _f(p[3])
                    nums = [_f(x) for x in p[4:4 + 2 * n]]
                    raw = [(a, b, zt) for a, b in _chunks(nums, 2)]
                elif ip in (2, 3):
                    step = 3 if ip == 2 else 6
                    nums = [_f(x) for x in p[3:3 + step * n]]
                    raw = [(c[0], c[1], c[2]) for c in _chunks(nums, step)]
                if de.form in (20, 21, 31, 32, 33, 34, 35, 36, 37, 38, 40):
                    skip("copious data annotation form")
                    continue
                pts = transformed(de, raw)
                if pts:
                    curve_pts.extend(pts)
                    add_preview(pts)
            elif t == 116:
                v = [_f(x) for x in p[1:4]]
                q = transformed(de, [(v[0], v[1], v[2])])
                if q:
                    point_cache[de.seq] = q[0]
                    if de.subordinate == "00":
                        curve_pts.extend(q)
            elif t == 126:
                k, m = _int(p[1]), _int(p[2])
                start_w = 7 + (k + m + 2)
                start_cp = start_w + (k + 1)
                nums = [_f(x) for x in p[start_cp:start_cp + 3 * (k + 1)]]
                cps = [(c[0], c[1], c[2]) for c in _chunks(nums, 3)]
                if len(cps) != k + 1:
                    skip("B-spline curve with incomplete data")
                    continue
                pts = transformed(de, cps)
                if pts:
                    curve_pts.extend(pts)
                    spline_used = True
                    add_preview(pts)
            elif t == 128:
                k1, k2, m1, m2 = _int(p[1]), _int(p[2]), _int(p[3]), _int(p[4])
                s_start = 10
                t_start = s_start + (k1 + m1 + 2)
                w_start = t_start + (k2 + m2 + 2)
                cp_start = w_start + (k1 + 1) * (k2 + 1)
                ncp = (k1 + 1) * (k2 + 1)
                nums = [_f(x) for x in p[cp_start:cp_start + 3 * ncp]]
                cps = [(c[0], c[1], c[2]) for c in _chunks(nums, 3)]
                if len(cps) != ncp:
                    skip("B-spline surface with incomplete data")
                    continue
                pts = transformed(de, cps)
                if pts:
                    surf_pts.extend(pts)
            elif t == 502:
                n = _int(p[1])
                nums = [_f(x) for x in p[2:2 + 3 * n]]
                raw = [(c[0], c[1], c[2]) for c in _chunks(nums, 3)]
                pts = transformed(de, raw)
                if pts:
                    curve_pts.extend(pts)
            elif t == 192:
                loc_ptr, axis_ptr, radius = _int(p[1]), _int(p[2]), _f(p[3])
                res.cylinders.append({"loc_ptr": loc_ptr, "axis_ptr": axis_ptr, "radius_model": radius, "de": de.seq})
        except Exception as exc:  # noqa: BLE001 - one bad entity must not kill the file
            skip(f"malformed entity {t} ({ENTITY_NAMES.get(t, 'unknown')})")
            log.debug("entity %s DE %s failed: %s", t, de.seq, exc)

    # annotation text (general notes) — useful for part-number matching only
    for de in des.values():
        if de.type == 212:
            try:
                p = get_params(de)
                ns = _int(p[1])
                for i in range(ns):
                    txt = p[2 + i * 12 + 11] if len(p) > 2 + i * 12 + 11 else None
                    if isinstance(txt, str) and txt.strip():
                        res.texts.append(txt.strip())
            except Exception:  # noqa: BLE001
                pass

    # resolve 192 cylinders (location point 116, axis direction 123)
    for cyl in res.cylinders:
        try:
            loc_de, ax_de = des.get(cyl["loc_ptr"]), des.get(cyl["axis_ptr"])
            loc = None
            if loc_de is not None and loc_de.type == 116:
                v = [_f(x) for x in get_params(loc_de)[1:4]]
                loc = transformed(loc_de, [(v[0], v[1], v[2])])[0]
            axis = (0.0, 0.0, 1.0)
            if ax_de is not None and ax_de.type == 123:
                v = [_f(x) for x in get_params(ax_de)[1:4]]
                axis = _norm3((v[0], v[1], v[2]))
            cyl["center"], cyl["normal"], cyl["radius"] = loc, axis, cyl["radius_model"] * to_mm
        except Exception:  # noqa: BLE001
            cyl["center"] = None

    # ---------------- bounding box
    def bbox(pts: List[Vec]) -> Optional[Tuple[Vec, Vec]]:
        if not pts:
            return None
        xs, ys, zs = zip(*pts)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    cb = bbox(curve_pts)
    sb = bbox(surf_pts)
    res.surface_bbox = sb
    if cb:
        res.bbox_min, res.bbox_max = cb
        res.bbox_kind = ("from edges/curves; B-spline control points included, so this is a conservative upper bound"
                         if spline_used else "from line end points, arcs and vertices (exact for this geometry)")
    elif sb:
        res.bbox_min, res.bbox_max = sb
        res.bbox_kind = "from B-spline surface control points — conservative upper bound (trimming not applied)"
    if res.bbox_min:
        res.extents_mm = [res.bbox_max[i] - res.bbox_min[i] for i in range(3)]
    res.preview = preview

    # ---------------- candidate holes from circles grouped by axis
    groups: Dict[Tuple, Dict[float, float]] = defaultdict(lambda: defaultdict(float))
    group_circles: Dict[Tuple, int] = defaultdict(int)
    for c in circles:
        if c["sweep_deg"] < 170:
            continue
        n = _canon_axis(c["normal"])
        d = round(2 * c["radius"], 3)
        ctr = c["center"]
        h = _dot(ctr, n)
        perp = (ctr[0] - h * n[0], ctr[1] - h * n[1], ctr[2] - h * n[2])
        key = (round(d, 2), tuple(round(x, 3) for x in n), tuple(round(x, 2) for x in perp))
        groups[key][round(h, 2)] += c["sweep_deg"]
        group_circles[key] += 1
    per_dia: Dict[float, Dict[str, int]] = defaultdict(lambda: {"features": 0, "circles": 0})
    for key, levels in groups.items():
        if max(levels.values()) >= 355:
            per_dia[key[0]]["features"] += 1
            per_dia[key[0]]["circles"] += group_circles[key]
    res.holes = [{"diameter_mm": d, "count": v["features"], "circles": v["circles"]}
                 for d, v in sorted(per_dia.items())]
    # cylinders (5.3 analytic surfaces) grouped the same way
    cyl_groups = defaultdict(int)
    for cyl in res.cylinders:
        if cyl.get("center") is None:
            continue
        n = _canon_axis(cyl["normal"])
        ctr = cyl["center"]
        h = _dot(ctr, n)
        perp = tuple(round(ctr[i] - h * n[i], 2) for i in range(3))
        cyl_groups[(round(2 * cyl["radius"], 2), tuple(round(x, 3) for x in n), perp)] += 1
    cyl_per_dia = Counter(k[0] for k in cyl_groups)

    # ---------------- model type
    if cnt.get(186):
        res.model_type = "solid B-rep (manifold solid B-rep object, entity 186)"
    elif any(cnt.get(t) for t in (150, 152, 154, 156, 158, 160, 162, 164, 168, 180)):
        res.model_type = "CSG solid primitives"
    elif any(cnt.get(t) for t in (144, 143, 128, 114, 118, 120, 122, 190, 192, 194, 196, 198)) or cnt.get(510):
        res.model_type = "surface model (no closed solid entity)"
    elif any(cnt.get(t) for t in CURVE_TYPES):
        res.model_type = "wireframe (curves only, no surfaces)"
    else:
        res.model_type = "no geometry entities found"
    if cnt.get(408) or cnt.get(412) or cnt.get(414):
        res.warnings.append("subfigure instances present but not expanded — bounding box may be incomplete")
    if res.transforms_failed:
        res.warnings.append(f"{res.transforms_failed} entities skipped because their transform could not be read")

    # ---------------- part numbers seen in metadata (file name / product id / start section / notes)
    meta_text = " ".join(str(x) for x in [res.global_params.get("product_id_sender"), res.global_params.get("file_name"),
                                          res.global_params.get("product_id_receiver"), res.start_text] + res.texts if x)
    res.part_numbers_seen = list(dict.fromkeys(re.findall(r"(?<![0-9A-Za-z.])([A-Za-z]{0,4}[-_]?\d{4,}[A-Za-z0-9\-]*)", meta_text)))[:20]
    res.facts = _facts(res, cyl_per_dia, scale)
    return res


def _compose(a: Matrix, b: Matrix) -> Matrix:
    """a ∘ b  (apply b first, then a)."""
    r = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(sum(a[i][k] * b[k][j] for k in range(3)))
        row.append(sum(a[i][k] * b[k][3] for k in range(3)) + a[i][3])
        r.append(tuple(row))
    return tuple(r)  # type: ignore[return-value]


def _facts(res: IgesResult, cyl_per_dia: Counter, scale: float) -> List[Fact]:
    F: List[Fact] = []
    fn = res.filename

    def add(category: str, field_name: str, value: str, location: str, unit: Optional[str] = None,
            num: Optional[float] = None, confidence: str = "high", meta: Optional[dict] = None) -> None:
        F.append(Fact(source_type="iges", source_file=fn, location=location, field=field_name, value=value,
                      unit=unit, num=num, confidence=confidence, category=category, raw=value, meta=meta or {}))

    gpar = res.global_params
    add("units", "Units", f"{res.units or 'unknown (assumed mm)'} ({res.unit_note})", "Global section, fields 14-15",
        confidence="high" if res.units else "low", meta={"unit": res.units})
    if scale != 1.0:
        add("model_scale", "Model space scale", fmt_num(scale), "Global section, field 13",
            meta={"scale": scale})
    for key, label in (("product_id_sender", "Product id"), ("file_name", "Original file name"),
                       ("native_system", "Created by system"), ("author", "Author"), ("organization", "Organization"),
                       ("file_date", "File date"), ("model_date", "Model date")):
        v = gpar.get(key)
        if isinstance(v, str) and v.strip():
            add("meta", label, v.strip(), "Global section")
    if res.total_entities:
        top = sorted(res.entity_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        summary = "; ".join(f"{t} {ENTITY_NAMES.get(t, '?')} ×{n}" for t, n in top[:14])
        add("entity_count", "Entity counts", f"{res.total_entities} entities: {summary}", "Directory section",
            meta={"counts": res.entity_counts})
    add("model_type", "Model type", res.model_type, "Directory section (entity types)")
    if res.extents_mm:
        ex = res.extents_mm
        conf = "high" if "exact" in res.bbox_kind else "medium"
        add("bbox", "Bounding box (X × Y × Z)",
            f"{fmt_num(ex[0])} × {fmt_num(ex[1])} × {fmt_num(ex[2])} mm ({res.bbox_kind})",
            "geometry, all units converted to mm", unit="mm", confidence=conf,
            meta={"extents_mm": ex, "min": res.bbox_min, "max": res.bbox_max,
                  "sorted": sorted(ex, reverse=True), "upper_bound": "upper bound" in res.bbox_kind})
        for axis, v in zip("XYZ", ex):
            add("extent", f"{axis} extent", f"{fmt_num(v)} mm", f"bounding box, {axis} axis", unit="mm", num=v,
                confidence=conf)
        if res.units and res.units != "mm":
            raw = [v / UNIT_TO_MM[res.units] * scale for v in ex]
            add("extent_native", "Extents in file units",
                f"{fmt_num(raw[0], 4)} × {fmt_num(raw[1], 4)} × {fmt_num(raw[2], 4)} {res.units} "
                f"(converted ×{UNIT_TO_MM[res.units]} to mm)", "bounding box before unit conversion", unit=res.units)
    else:
        add("bbox", "Bounding box", "not available (no supported geometry found)", "geometry", confidence="low")
    if res.surface_bbox and res.extents_mm and "surface" not in res.bbox_kind:
        smin, smax = res.surface_bbox
        se = [smax[i] - smin[i] for i in range(3)]
        if any(abs(se[i] - res.extents_mm[i]) > 0.5 for i in range(3)):
            add("bbox_surface", "Surface control-point box (upper bound)",
                f"{fmt_num(se[0])} × {fmt_num(se[1])} × {fmt_num(se[2])} mm (untrimmed surfaces; larger than the edge box)",
                "B-spline surface control points", unit="mm", confidence="low")
    for h in res.holes:
        if h["count"] <= 0:
            continue
        add("candidate_hole", "Candidate hole feature",
            f"Ø{fmt_num(h['diameter_mm'])} mm × {h['count']} (circular edges grouped by axis: {h['circles']} circle "
            f"entities; may also be bosses or outer diameters)", "circular arcs (entity 100)", unit="mm",
            num=h["diameter_mm"], confidence="medium",
            meta={"diameter": h["diameter_mm"], "count": h["count"], "circles": h["circles"]})
    for d, n in sorted(cyl_per_dia.items()):
        add("candidate_hole", "Cylindrical surface",
            f"Ø{fmt_num(d)} mm × {n} (right circular cylindrical surfaces grouped by axis; holes or shafts)",
            "entity 192", unit="mm", num=d, confidence="medium",
            meta={"diameter": d, "count": n, "from": "cylinder"})
    add("volume_na", "Volume / mass / surface area",
        "not reliably available from this IGES (no verified closed-solid evaluation is done)", "by design",
        confidence="high")
    if res.transforms_applied:
        add("transform_note", "Transformations", f"entity-124 transforms applied to {res.transforms_applied} entities",
            "Directory field 7")
    if res.skipped:
        total = sum(res.skipped.values())
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(res.skipped.items(), key=lambda kv: -kv[1])[:6])
        add("skipped", "Entities skipped", f"{total} ({detail})", "parser report", confidence="high")
    for t in res.texts[:5]:
        add("model_text", "Text note in model", t, "general note entity 212", confidence="medium")
    return F
