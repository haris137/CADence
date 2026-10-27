"""Drafting-table theme for Streamlit + small HTML renderers (citation chips, check rows, source cards).

Fonts load from Google Fonts when online; every family has an offline fallback so a dead venue
network only changes the typeface, never the layout.
"""
from __future__ import annotations

import html
import re
from typing import Dict, Iterable, List, Optional

from .facts import Fact

SOURCE_NAMES = {"E": "Excel", "P": "Drawing", "I": "3D model", "C": "Check", "M": "Missing"}

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --paper: #F2F1E8;
  --paper-2: #FAF9F3;
  --grid: rgba(20, 49, 92, 0.055);
  --grid-major: rgba(20, 49, 92, 0.10);
  --ink: #14315C;
  --ink-soft: #52637E;
  --ink-faint: #8A97AB;
  --rule: rgba(20, 49, 92, 0.28);
  --panel: #E6ECF2;
  --accent: #F2611D;
  --match: #2F7A4F;
  --mismatch: #B3261E;
  --nc: #8A8F98;
  --info: #3D6FA8;
  --e: #2D6A4F; --p: #8E3B46; --i: #1F5FA8; --c: #B4500F; --m: #6B7280;
  --display: 'Barlow Condensed', 'Arial Narrow', 'Roboto Condensed', 'Segoe UI', sans-serif;
  --body: 'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
  --mono: 'IBM Plex Mono', Consolas, 'Cascadia Mono', 'Courier New', monospace;
}

html, body, .stApp, [data-testid="stAppViewContainer"] { color: var(--ink); font-family: var(--body); }
.stApp {
  background-color: var(--paper);
  background-image:
    linear-gradient(var(--grid-major) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-major) 1px, transparent 1px),
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 120px 120px, 120px 120px, 24px 24px, 24px 24px;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1380px; }
h1, h2, h3, h4 { font-family: var(--display) !important; color: var(--ink) !important; letter-spacing: 0.01em; }
p, li, label, .stMarkdown { color: var(--ink); }
code { font-family: var(--mono); color: var(--ink); background: rgba(20,49,92,0.06); }

/* ---------- sidebar = title block / control panel ---------- */
[data-testid="stSidebar"] { background: var(--panel); border-right: 2px solid var(--ink); }
[data-testid="stSidebar"] > div:first-child { background: var(--panel); }
[data-testid="stSidebar"] * { color: var(--ink); }
.tb-head { border: 2px solid var(--ink); padding: 10px 12px 8px; margin-bottom: 14px; background: var(--paper-2); }
.tb-head .t { font-family: var(--display); font-size: 1.65rem; font-weight: 700; line-height: 1; }
.tb-head .s { font-size: 0.78rem; color: var(--ink-soft); margin-top: 4px; }
.tb-cell { border: 1px solid var(--ink); border-bottom: none; padding: 6px 10px 2px; background: var(--paper-2);
  font-family: var(--display); font-size: 1.02rem; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { border: 1px solid var(--ink); border-radius: 0; background: var(--paper-2); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-family: var(--display); font-size: 1.02rem; font-weight: 600; }
.filelist { font-family: var(--mono); font-size: 0.72rem; line-height: 1.45; color: var(--ink-soft); margin: 4px 0 8px; }
.filelist b { color: var(--ink); font-weight: 500; }
.filelist .skip { color: var(--mismatch); }

/* ---------- inputs & buttons ---------- */
.stButton button p, .stDownloadButton button p { font-family: var(--display) !important; font-size: 1.05rem !important; font-weight: 600; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
  border-radius: 2px; border: 1.5px solid var(--ink); background: var(--paper-2); color: var(--ink);
  font-family: var(--display); font-weight: 600; font-size: 1.02rem; letter-spacing: 0.02em; box-shadow: 2px 2px 0 var(--ink);
  transition: transform .06s ease, box-shadow .06s ease;
}
.stButton button:hover { border-color: var(--ink); color: var(--ink); background: #fff; }
.stButton button:active { transform: translate(2px, 2px); box-shadow: 0 0 0 var(--ink); }
.stButton button[kind="primary"], button[data-testid="stBaseButton-primary"] { background: var(--accent); border-color: var(--ink); color: #fff; }
.stButton button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: #fff; }
.stButton button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #E2520F; color: #fff; }
.stButton button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { border-radius: 2px !important; }
input, textarea { font-family: var(--mono) !important; }

/* ---------- tabs (old baseweb markup and the newer react-aria markup) ---------- */
[data-baseweb="tab-list"], [role="tablist"] { gap: 0 !important; border-bottom: 2px solid var(--ink); }
[data-baseweb="tab"], [data-testid="stTab"] { padding: 6px 20px 4px !important; border: 1.5px solid transparent; border-bottom: none; }
[data-baseweb="tab"] p, [data-testid="stTab"] p { font-family: var(--display) !important; font-size: 1.22rem !important; font-weight: 600 !important; color: var(--ink-soft); }
[data-baseweb="tab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] { border-color: var(--ink); background: var(--paper-2); }
[data-testid="stTab"][aria-selected="true"] p, [data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); }
[data-baseweb="tab-highlight"], [data-testid="stTab"] .react-aria-SelectionIndicator { background: var(--accent) !important; height: 3px; }

/* ---------- hero: drawing title ---------- */
.hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 16px; }
.hero .pn { font-family: var(--display); font-weight: 700; font-size: 5.2rem; line-height: 0.82; color: var(--ink); }
.hero .pn small { font-size: 1.1rem; font-weight: 500; color: var(--ink-soft); display: block; margin-bottom: 8px; letter-spacing: 0.04em; }
.hero .desc { font-family: var(--display); font-size: 1.7rem; font-weight: 500; color: var(--ink-soft); margin-top: 6px; }
.hero .meta { font-size: 0.86rem; color: var(--ink-soft); margin-top: 6px; }
.stamp { border: 3px solid var(--accent); color: var(--accent); padding: 6px 16px 5px; transform: rotate(-3deg);
  font-family: var(--display); text-align: center; background: rgba(242, 97, 29, 0.04); min-width: 190px; max-width: 280px; }
.stamp .lv { font-size: 2.3rem; font-weight: 700; line-height: 1; }
.stamp .sub { font-size: 0.92rem; font-weight: 600; }
.stamp.dim { border-color: var(--nc); color: var(--nc); }

/* ---------- source cards ---------- */
.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
.card { border: 1.5px solid var(--ink); background: var(--paper-2); padding: 10px 14px 10px 18px; position: relative; min-height: 112px; }
.card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--nc); }
.card.loaded::before { background: var(--match); }
.card.error::before, .card.timeout::before { background: var(--mismatch); }
.card.no_match::before, .card.not_found::before, .card.empty::before { background: var(--accent); }
.card .k { font-family: var(--display); font-size: 1.35rem; font-weight: 700; }
.card .st { font-family: var(--mono); font-size: 0.76rem; float: right; padding: 1px 6px; border: 1px solid currentColor; }
.card.loaded .st { color: var(--match); } .card.error .st, .card.timeout .st { color: var(--mismatch); }
.card.missing .st { color: var(--nc); } .card.no_match .st, .card.not_found .st, .card.empty .st { color: var(--accent); }
.card .f { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-soft); margin-top: 4px; word-break: break-all; }
.card .n { font-size: 0.86rem; margin-top: 6px; }
.card .msg { font-size: 0.76rem; color: var(--ink-soft); margin-top: 4px; }

/* ---------- cross-check rows ---------- */
.checks { border: 1.5px solid var(--ink); background: var(--paper-2); }
.chk { display: grid; grid-template-columns: 128px 1fr; gap: 10px; padding: 8px 12px; border-bottom: 1px solid var(--rule); }
.chk:last-child { border-bottom: none; }
.pill { font-family: var(--mono); font-size: 0.72rem; font-weight: 500; text-align: center; padding: 3px 4px; border: 1.5px solid currentColor; align-self: start; }
.chk.MATCH .pill { color: var(--match); } .chk.MISMATCH .pill { color: #fff; background: var(--mismatch); border-color: var(--mismatch); }
.chk.NOT.COMPARABLE .pill, .chk.NC .pill { color: var(--nc); } .chk.INFO .pill { color: var(--info); }
.chk.MISMATCH { background: rgba(179, 38, 30, 0.06); }
.chk .nm { font-family: var(--display); font-weight: 600; font-size: 1.08rem; line-height: 1.1; }
.chk .dt { font-size: 0.86rem; color: var(--ink-soft); margin-top: 2px; }
.summary { font-family: var(--display); font-size: 1.15rem; margin: 2px 0 8px; }
.summary b.bad { color: var(--mismatch); } .summary b.ok { color: var(--match); }

/* ---------- tags & evidence chips ---------- */
.tag { font-family: var(--mono); font-size: 0.78em; font-weight: 500; padding: 0 4px; border: 1px solid currentColor; border-radius: 2px; white-space: nowrap; }
.tag.E { color: var(--e); } .tag.P { color: var(--p); } .tag.I { color: var(--i); } .tag.C { color: var(--c); } .tag.M { color: var(--m); } .tag.X { color: var(--mismatch); text-decoration: line-through; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
details.chip { font-family: var(--mono); font-size: 0.74rem; border: 1px solid currentColor; background: var(--paper-2); border-radius: 2px; max-width: 100%; }
details.chip summary { cursor: pointer; padding: 2px 7px; list-style: none; }
details.chip summary::-webkit-details-marker { display: none; }
details.chip[open] { flex-basis: 100%; }
details.chip .ev { color: var(--ink); padding: 6px 9px 8px; border-top: 1px dashed currentColor; line-height: 1.5; white-space: normal; }
details.chip .ev .lbl { color: var(--ink-faint); }
details.chip.E { color: var(--e); } details.chip.P { color: var(--p); } details.chip.I { color: var(--i); } details.chip.C { color: var(--c); } details.chip.M { color: var(--m); }
.badge { font-family: var(--mono); font-size: 0.72rem; color: var(--ink-soft); margin-top: 6px; }
.badge .warn { color: #fff; background: var(--accent); padding: 1px 6px; }
.badge .nollm { color: var(--info); border: 1px solid var(--info); padding: 0 5px; }
.note { font-size: 0.8rem; color: var(--ink-soft); margin-top: 4px; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: var(--paper-2); border: 1px solid var(--rule); border-radius: 2px; }
[data-testid="stChatInput"] textarea { font-family: var(--body) !important; }

/* ---------- empty states ---------- */
.empty { border: 1.5px dashed var(--ink); background: rgba(250, 249, 243, 0.85); padding: 28px 30px; max-width: 760px; }
.empty h2 { margin: 0 0 6px; font-size: 2.4rem; }
.empty ol { margin: 10px 0 0 18px; } .empty li { margin: 4px 0; }
.nf { border: 2px solid var(--accent); background: var(--paper-2); padding: 20px 24px; max-width: 820px; }
.nf .big { font-family: var(--display); font-size: 2.6rem; font-weight: 700; line-height: 1; }

@media (max-width: 900px) {
  .cards { grid-template-columns: 1fr; }
  .hero { flex-direction: column; align-items: flex-start; }
  .hero .pn { font-size: 3.6rem; }
}
@media (prefers-reduced-motion: reduce) { .stButton button { transition: none; } }
"""


def inject(st) -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s), quote=True)


_TAG_GROUP = re.compile(r"\[([A-Za-z]{1,3}\d+(?:\s*[,;/]\s*[A-Za-z]{1,3}\d+)*)\]")


def tagify(text: str, known: Optional[Iterable[str]] = None) -> str:
    """Escape text, then turn [E3] / [P1, I2] citations into coloured inline tags. Unknown tags are struck out."""
    known_set = set(known) if known is not None else None
    safe = esc(text)

    def repl(m: re.Match) -> str:
        parts = re.findall(r"[A-Za-z]{1,3}\d+", m.group(1))
        out = []
        for t in parts:
            t = t.upper()
            cls = t[0] if t[0] in "EPICM" else "X"
            if known_set is not None and t not in known_set:
                cls = "X"
            out.append(f'<span class="tag {cls}" title="{esc(SOURCE_NAMES.get(t[0], "unknown"))}">{t}</span>')
        return " ".join(out)

    return _TAG_GROUP.sub(repl, safe)


def md_light(text: str, known: Optional[Iterable[str]] = None) -> str:
    """Minimal, safe markdown: **bold**, bullet lists, line breaks — after escaping."""
    lines = tagify(text, known).split("\n")
    out, in_list = [], False
    for ln in lines:
        ln = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ln)
        if re.match(r"^\s*[-*•]\s+", ln):
            if not in_list:
                out.append("<ul style='margin:4px 0 4px 18px;padding:0'>")
                in_list = True
            out.append("<li>" + re.sub(r"^\s*[-*•]\s+", "", ln) + "</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(ln + "<br>" if ln.strip() else "")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def chips(tags: Iterable[str], by_id: Dict[str, Fact]) -> str:
    parts = []
    for t in tags:
        f = by_id.get(t)
        if f is None:
            parts.append(f'<details class="chip M"><summary>{esc(t)} · not a real fact</summary>'
                         f'<div class="ev">The answer cited a tag that does not exist.</div></details>')
            continue
        src = SOURCE_NAMES.get(t[0], "")
        file_line = f'<span class="lbl">file</span> {esc(f.source_file)}<br>' if f.source_file else ""
        conf = f'<br><span class="lbl">confidence</span> {esc(f.confidence)}' if f.confidence != "high" else ""
        raw = f'<br><span class="lbl">raw</span> {esc(f.raw[:300])}' if f.raw and f.raw != f.value else ""
        parts.append(
            f'<details class="chip {t[0]}"><summary>{esc(t)} · {esc(src)} · {esc(f.field[:28])}</summary>'
            f'<div class="ev">{file_line}<span class="lbl">where</span> {esc(f.location)}<br>'
            f'<span class="lbl">value</span> {tagify(f.short_value(400))}{(" " + esc(f.unit)) if f.unit and f.source_type not in ("check", "missing") else ""}'
            f'{raw}{conf}</div></details>')
    return f'<div class="chips">{"".join(parts)}</div>' if parts else ""


def check_rows(checks, known: Iterable[str]) -> str:
    rows = []
    for c in checks:
        cls = "NC" if c.status == "NOT COMPARABLE" else c.status
        rows.append(f'<div class="chk {cls}"><div class="pill">{esc(c.status)}</div>'
                    f'<div><div class="nm">{esc(c.name)}</div><div class="dt">{tagify(c.detail, known)}</div></div></div>')
    return f'<div class="checks">{"".join(rows)}</div>' if rows else ""


STATE_TEXT = {"loaded": "loaded", "missing": "not provided", "no_match": "no matching file", "not_found": "part not in file",
              "error": "read error", "timeout": "timed out", "empty": "no usable data"}


def source_cards(status: Dict, names=("excel", "pdf", "iges")) -> str:
    titles = {"excel": "Excel record", "pdf": "PDF drawing", "iges": "IGES model"}
    out = []
    for k in names:
        s = status.get(k)
        if s is None:
            continue
        files = ", ".join(s.files[:3]) + (f" +{len(s.files) - 3}" if len(s.files) > 3 else "") if s.files else "—"
        n = (f"<b>{s.n_facts}</b> facts · {s.seconds:.2f} s" if s.state == "loaded" else "")
        msg = esc(s.message[:180]) if s.message and s.state != "loaded" else ""
        out.append(f'<div class="card {s.state}"><span class="st">{esc(STATE_TEXT.get(s.state, s.state))}</span>'
                   f'<div class="k">{titles[k]}</div><div class="f">{esc(files)}</div>'
                   f'<div class="n">{n}</div><div class="msg">{msg}</div></div>')
    return f'<div class="cards">{"".join(out)}</div>'
