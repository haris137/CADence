"""Unit handling and number parsing. All arithmetic lives here, never in the LLM."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

UNIT_TO_MM: Dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "km": 1_000_000.0,
    "in": 25.4,
    "ft": 304.8,
    "mi": 1_609_344.0,
    "mil": 0.0254,
    "um": 0.001,
    "uin": 0.0000254,
}

# IGES 5.3 Global section field 14 (units flag)
IGES_UNIT_FLAGS: Dict[int, Optional[str]] = {
    1: "in", 2: "mm", 3: None, 4: "ft", 5: "mi", 6: "m", 7: "km", 8: "mil", 9: "um", 10: "cm", 11: "uin",
}

_UNIT_SYNONYMS = {
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm", "mms": "mm",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "km": "km",
    "in": "in", "inch": "in", "inches": "in", "\"": "in", "ins": "in", "''": "in",
    "ft": "ft", "foot": "ft", "feet": "ft", "'": "ft",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "mil": "mil", "mils": "mil", "thou": "mil",
    "um": "um", "micron": "um", "microns": "um", "µm": "um", "μm": "um", "micrometer": "um", "micrometre": "um",
    "uin": "uin", "microinch": "uin", "microinches": "uin", "µin": "uin",
}

NUM_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


def normalize_unit(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip().lower().strip(".()[]{} ")
    if not t:
        return None
    if t in _UNIT_SYNONYMS:
        return _UNIT_SYNONYMS[t]
    # phrases like "dimensions in mm", "all dimensions are in millimeters"
    m = re.search(r"\b(millimet(?:er|re)s?|mm|inch(?:es)?|in|centimet(?:er|re)s?|cm|met(?:er|re)s?|microns?|feet|ft)\b", t)
    if m:
        return _UNIT_SYNONYMS.get(m.group(1), None) or normalize_unit(m.group(1))
    return None


def parse_number(text) -> Optional[float]:
    """Parse numbers like '12', '-.25', '1.5D0', '2.0E-3', '+3.', '2280.0'. Returns None if not a number."""
    if text is None:
        return None
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(float(text)) else None
    s = str(text).strip().replace(" ", "")
    if not s:
        return None
    s = s.replace("D", "E").replace("d", "e")
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def to_mm(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or unit is None:
        return None
    f = UNIT_TO_MM.get(unit)
    return None if f is None else value * f


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "?"
    if abs(x - round(x)) < 10 ** (-digits):
        return str(int(round(x)))
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return s


def fmt_mm(x: Optional[float]) -> str:
    return f"{fmt_num(x)} mm"


def close(a: float, b: float, rel: float = 0.01, abs_tol: float = 0.5) -> bool:
    """Values agree if within the absolute OR the relative tolerance."""
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


_DIM_SPLIT = re.compile(r"\s*(?:x|X|×|\*|by|BY)\s*")
_ROUND_MARK = re.compile(r"(Ø|⌀|∅|%%c|\bdia\.?\b|\bdiam\.?\b|\bod\b|\brd\b|\bround\b|\bbar\b)", re.IGNORECASE)


def parse_dims(text: Optional[str]) -> Optional[dict]:
    """Parse a size string such as '100 x 60 x 12 mm', 'Ø40 x 160', '4.0 X 2.5 X .5 IN', 'DIA 45 X 155 LG'.

    Returns {"values": [...], "unit": str|None, "round": bool, "triplet": [a, b, c] sorted desc}
    or None when fewer than 2 numbers are present.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    unit = None
    um = re.search(r"(mm|millimet(?:er|re)s?|cm|inch(?:es)?|in\b|\"|metres?|meters?)\s*$", s, re.IGNORECASE)
    if um:
        unit = normalize_unit(um.group(1))
    if unit is None:
        um2 = re.search(r"\b(mm|inch(?:es)?|cm)\b", s, re.IGNORECASE)
        if um2:
            unit = normalize_unit(um2.group(1))
    is_round = bool(_ROUND_MARK.search(s))
    # only treat as a size string if numbers are separated by x / × / by
    if not re.search(r"\d\s*(?:x|X|×|\*|by|BY)\s*[Ø⌀∅]?\s*[-+.]?\d", s):
        return None
    nums = [parse_number(n) for n in NUM_RE.findall(s.replace("Ø", " ").replace("⌀", " ").replace("∅", " "))]
    nums = [n for n in nums if n is not None and n > 0]
    if len(nums) < 2:
        return None
    nums = nums[:3]
    if is_round and len(nums) == 2:
        d, l = nums[0], nums[1]
        # 'Ø40 x 160' -> diameter then length (the diameter is marked); bounding triplet is (L, D, D)
        triplet = sorted([l, d, d], reverse=True)
    elif len(nums) == 2:
        triplet = sorted(nums, reverse=True)
    else:
        triplet = sorted(nums, reverse=True)
    return {"values": nums, "unit": unit, "round": is_round, "triplet": triplet}


def numbers_in(text: str) -> List[float]:
    out = []
    for n in NUM_RE.findall(text or ""):
        v = parse_number(n)
        if v is not None:
            out.append(v)
    return out


def iges_unit(flag: Optional[int], name: Optional[str]) -> Tuple[Optional[str], str]:
    """Resolve IGES units from flag (field 14) and name (field 15). Returns (unit, note)."""
    by_flag = IGES_UNIT_FLAGS.get(flag) if flag is not None else None
    by_name = None
    if name:
        n = name.strip().upper().strip("'\"")
        by_name = {"IN": "in", "INCH": "in", "INCHES": "in", "MM": "mm", "FT": "ft", "MI": "mi", "M": "m",
                   "KM": "km", "MIL": "mil", "UM": "um", "MICRON": "um", "CM": "cm", "UIN": "uin",
                   "MICROINCH": "uin"}.get(n)
    if by_flag and by_name and by_flag != by_name:
        return by_flag, f"units flag {flag} ({by_flag}) disagrees with unit name '{name}'; using the flag as the IGES spec requires"
    if by_flag:
        return by_flag, f"units flag {flag}"
    if flag == 3 and by_name:
        return by_name, f"units flag 3 (named units) -> '{name}'"
    if by_name:
        return by_name, f"unit name '{name}' (flag {flag} not recognised)"
    return None, f"units not specified (flag {flag}, name {name!r})"
