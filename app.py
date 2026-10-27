"""CNC Part Analyzer — Streamlit app (SPEC'26 LLM-based CNC part analysis).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional

import streamlit as st

st.set_page_config(page_title="CADence", page_icon="📐", layout="wide", initial_sidebar_state="expanded")

# heavy imports up front so the first question is not slowed down by them
import pandas as pd  # noqa: E402

try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import pipeline as PL  # noqa: E402
from src import qa, ui_theme  # noqa: E402
from src.excel_parser import normalize_pn  # noqa: E402
from src.units import fmt_num  # noqa: E402
from src.llm import LLMClient, providers_from_env  # noqa: E402

SAMPLES_DIR = os.path.join(ROOT, "samples")

# Streamlit >= 1.49 sizes widgets with width="stretch"; older versions use use_container_width=True
try:
    _ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:  # pragma: no cover
    _ST_VER = (1, 0)
STRETCH = {"width": "stretch"} if _ST_VER >= (1, 49) else {"use_container_width": True}
SUGGESTED = [("Material", "What material is it?"), ("Overall dimensions", "What are the overall dimensions?"),
             ("Holes", "What holes does it have?"), ("Agreement", "Do the files agree?"),
             ("Operations", "List the machining operations"), ("Price", "What's the price?")]
AVATAR_USER = ":material/person:" if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 40) else "🧑"
AVATAR_BOT = ":material/precision_manufacturing:" if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 40) else "🛠️"

ui_theme.inject(st)
ss = st.session_state
for key, default in {"files": [], "files_label": "", "analysis": None, "chats": {}, "llm_log": [],
                     "upload_sig": None, "known": [], "known_sig": None, "debug": []}.items():
    ss.setdefault(key, default)
if "tmpdir" not in ss:
    ss.tmpdir = tempfile.mkdtemp(prefix="cnc_session_")
if "_next_part" in ss:                       # set by buttons before the text box exists
    ss.part_input = ss.pop("_next_part")
    ss._auto_analyze = True
if "_sync_part" in ss:                       # a question switched the part: keep the sidebar box in step
    ss.part_input = ss.pop("_sync_part")


# ============================================================================================ helpers

def set_items(items: List[PL.FileItem], label: str) -> None:
    ss.files = items
    ss.files_label = label
    ss.analysis = None
    ss.known_sig = None


def save_to_tmp(name: str, data: bytes) -> None:
    """Uploads are kept only in this session's temp folder (deleted on reset)."""
    try:
        with open(os.path.join(ss.tmpdir, os.path.basename(name)), "wb") as fh:
            fh.write(data)
    except OSError:
        pass


def known_parts() -> List[str]:
    sig = tuple((i.name, len(i.data)) for i in ss.files)
    if ss.known_sig != sig:
        try:
            ss.known = PL.known_part_numbers(ss.files)
        except Exception:
            ss.known = []
        ss.known_sig = sig
    return ss.known


def log_llm(llm: Optional[LLMClient]) -> None:
    if llm is not None and llm.log:
        ss.llm_log = (ss.llm_log + llm.log)[-60:]
        llm.log = []


def run_analysis(pn: str, settings: PL.Settings, llm: Optional[LLMClient]) -> Optional[PL.PartAnalysis]:
    pn = normalize_pn(pn)
    if not pn:
        st.sidebar.warning("Type a part number first.")
        return None
    with st.status(f"Analyzing part {pn}…", expanded=True) as status:
        def say(msg: str) -> None:
            status.update(label=msg)
            st.write(msg)
        try:
            pa = PL.analyze_part(pn, ss.files, settings, llm, progress=say)
        except Exception as exc:  # never show a stack trace to the user
            ss.debug.append(traceback.format_exc())
            status.update(label=f"Could not analyze {pn}: {exc}", state="error")
            return None
        label = f"Part {pn}: {pa.level_label} — {pa.seconds:.1f} s" if pa.found else f"Part {pn} is not in the provided files"
        status.update(label=label, state="complete" if pa.found else "error", expanded=False)
    log_llm(llm)
    ss.analysis = pa
    return pa


def describe(pa: PL.PartAnalysis) -> str:
    for cat in ("description", "title"):
        f = next((f for f in pa.facts if f.category == cat), None)
        if f:
            return f.short_value(60)
    return ""


def stamp_sub(pa: PL.PartAnalysis) -> str:
    if pa.level == 3:
        return "3 sources cross-checked"
    names = {"excel": "Excel", "pdf": "drawing", "iges": "3D model"}
    missing = [names[k] for k in ("excel", "pdf", "iges") if k not in pa.loaded]
    return "missing: " + ", ".join(missing) if missing else ""


def history_for_llm(chat: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for turn in chat[-2:]:
        out.append({"role": "user", "content": turn["q"]})
        out.append({"role": "assistant", "content": turn["ans"].text})
    return out


def render_answer(ans: qa.Answer, pa: PL.PartAnalysis) -> None:
    by_id = pa.by_id()
    st.markdown(f'<div class="ans">{ui_theme.md_light(ans.text, by_id.keys())}</div>', unsafe_allow_html=True)
    tags = ans.cited[:14]
    extra = len(ans.cited) - len(tags)
    if tags:
        st.markdown(ui_theme.chips(tags, by_id) + (f'<div class="badge">+{extra} more tags in the answer</div>' if extra > 0 else ""),
                    unsafe_allow_html=True)
    if ans.mode == "llm":
        b = f"answered by {ui_theme.esc(ans.provider)} from {ans.facts_sent} facts, {ans.llm_calls} LLM call(s)"
        if not ans.verified:
            b = (f'<span class="warn">unverified</span> {b} — {ui_theme.esc("; ".join(ans.issues))}')
        else:
            b += ", citations verified"
    elif ans.mode == "template":
        b = '<span class="nollm">answered without LLM</span> deterministic template built from the facts'
    elif ans.mode == "not_found":
        b = "no LLM call — the files do not contain this"
    else:
        b = "no LLM call — refused"
    st.markdown(f'<div class="badge">{b}</div>', unsafe_allow_html=True)
    if ans.note:
        st.markdown(f'<div class="note">{ui_theme.esc(ans.note)}</div>', unsafe_allow_html=True)


def answer(q: str, pa: PL.PartAnalysis, llm: Optional[LLMClient], settings: PL.Settings) -> PL.PartAnalysis:
    """Answer one question inside the current chat column (streams when an LLM is configured)."""
    # a different part number in the question -> switch part first (works for any part in the files)
    others = [p for p in qa.extract_part_numbers(q) if normalize_pn(p).upper() != pa.part_number.upper()]
    switched = ""
    if others:
        new = run_analysis(others[0], settings, llm)
        if new is not None:
            pa = new
            ss._sync_part = new.part_number
            switched = f"Switched to part {new.part_number}."
    chat = ss.chats.setdefault(pa.part_number, [])
    with st.chat_message("user", avatar=AVATAR_USER):
        st.write(q)
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        early, info = qa.precheck(q, pa.facts, pa.part_number, pa.found)
        if early is not None:
            ans = early
        elif llm is None:
            ans = qa.template_answer(pa.facts, q, info)
        else:
            prep = qa.prepare(q, pa.facts, pa.part_number, history_for_llm(chat), info)
            before = llm.calls
            ph = st.empty()
            try:
                with ph.container():
                    text = st.write_stream(llm.stream(prep.messages))
                if not isinstance(text, str):
                    text = "".join(str(t) for t in text)
                ans = qa.finalize(text, prep, pa.facts, pa.part_number, llm, llm.last_provider or "", before)
            except Exception as exc:  # every provider failed -> deterministic answer, never a blank screen
                ans = qa.template_answer(pa.facts, q, info)
                ans.note = f"LLM unavailable ({str(exc)[:140]}). Answered without the LLM instead."
            ph.empty()
            log_llm(llm)
        if switched:
            ans.note = (switched + " " + ans.note).strip()
        render_answer(ans, pa)
    chat.append({"q": q, "ans": ans})
    return pa


# ============================================================================================ sidebar

with st.sidebar:
    st.markdown('<div class="tb-head"><div class="t">CADence</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="tb-cell">1 &nbsp;Files</div>', unsafe_allow_html=True)
    mode = st.radio("Input mode", ["Upload files", "Folder on this computer", "Sample dataset"], key="mode",
                    label_visibility="collapsed")
    if mode == "Upload files":
        ups = st.file_uploader("Excel/CSV, PDF and IGES files (any names, any order)", accept_multiple_files=True,
                               key="uploader")
        if ups:
            sig = tuple((u.name, u.size) for u in ups)
            if sig != ss.upload_sig:
                pairs = [(u.name, u.getvalue()) for u in ups]
                for n, d in pairs:
                    save_to_tmp(n, d)
                set_items(PL.make_items(pairs), "uploaded files")
                ss.upload_sig = sig
    elif mode == "Folder on this computer":
        folder = st.text_input("Folder path", placeholder=r"D:\SPEC26\part_files", key="folder")
        if st.button("Scan folder", **STRETCH):
            try:
                set_items(PL.scan_folder(folder), f"folder {folder}")
            except Exception as exc:
                st.error(f"Could not scan the folder: {exc}")
    else:
        sets = PL.sample_sets(SAMPLES_DIR)
        if not sets:
            st.info("The sample files have not been generated yet.")
            if st.button("Generate sample files", **STRETCH):
                try:
                    sys.path.insert(0, os.path.join(ROOT, "scripts"))
                    import make_samples  # noqa: WPS433
                    make_samples.main(SAMPLES_DIR)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sample generation failed: {exc}")
        else:
            choice = st.selectbox("Sample set", list(sets), key="sample_set")
            if st.button("Load sample set", **STRETCH):
                set_items(PL.load_sample_set(SAMPLES_DIR, choice), f"sample set: {choice}")

    if ss.files:
        lines = []
        for it in ss.files[:40]:
            kind = {"excel": "Excel", "pdf": "PDF", "iges": "IGES"}.get(it.kind or "", "")
            if it.kind:
                lines.append(f"<b>{ui_theme.esc(kind)}</b> {ui_theme.esc(it.name)}" +
                             (f" <span class='skip'>({ui_theme.esc(it.note)})</span>" if it.note else ""))
            else:
                lines.append(f"<span class='skip'>skipped {ui_theme.esc(it.name)}: {ui_theme.esc(it.note)}</span>")
        st.markdown(f"<div class='filelist'>{'<br>'.join(lines)}</div>", unsafe_allow_html=True)

    st.markdown('<div class="tb-cell">2 &nbsp;Part number</div>', unsafe_allow_html=True)
    pn_typed = st.text_input("Part number", key="part_input", placeholder="e.g. 2280", label_visibility="collapsed")
    pn_choice = None
    kp = known_parts() if ss.files else []
    if ss.files and not pn_typed.strip() and kp:
        pn_choice = st.selectbox("Parts found in the files — pick one to confirm", kp, index=None,
                                 placeholder="choose a part number")
    analyze_clicked = st.button("Analyze part", type="primary", **STRETCH, disabled=not ss.files)

    with st.expander("3  LLM provider", expanded=False):
        use_llm = st.toggle("Use the LLM", value=True,
                            help="Off = no-LLM mode: facts, cross-checks and template answers still work.")
        # API keys, models and Ollama are configured by the developers in the backend (.env), never in the UI.
        _cfg = providers_from_env()
        if _cfg:
            st.caption("Configured in the backend: " + " → ".join(p.label for p in _cfg))
        else:
            st.caption("No provider configured in the backend. The app runs in no-LLM mode.")
        test_clicked = st.button("Test connection", **STRETCH)

    with st.expander("4  Tolerances and limits", expanded=False):
        tol_pct = st.slider("Relative tolerance (%)", 0.1, 5.0, 1.0, 0.1)
        tol_abs = st.number_input("Absolute tolerance (mm)", 0.01, 5.0, 0.5, 0.05)
        timeout_s = st.slider("Per-file read timeout (s)", 2, 30, 6)
        structuring = st.toggle("Structure drawing text with the LLM when loading", value=True,
                                help="One LLM call per PDF at load time; every value is checked against the PDF text.")

    if st.button("Reset session", **STRETCH):
        shutil.rmtree(ss.get("tmpdir", ""), ignore_errors=True)
        PL.clear_caches()
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()

settings = PL.Settings(tol_rel=tol_pct / 100.0, tol_abs_mm=float(tol_abs), timeout_s=float(timeout_s),
                       use_llm=use_llm, llm_structuring=structuring)
providers = providers_from_env() if use_llm else []
llm: Optional[LLMClient] = LLMClient(providers, timeout=30, max_tokens=500) if providers else None

if test_clicked:
    with st.sidebar:
        if llm is None:
            st.warning("No provider configured (or the LLM is switched off). The app runs in no-LLM mode.")
        else:
            for r in llm.test_connection():
                (st.success if r["status"] == "ok" else st.error)(f"{r['provider']}: {r['status']} — {r['detail']}")

# ============================================================================================ main

def main() -> None:
    target = (pn_typed or "").strip() or (pn_choice or "")
    if analyze_clicked or ss.pop("_auto_analyze", False):
        if target:
            run_analysis(target, settings, llm)
        else:
            st.sidebar.warning("Type a part number (or pick one from the list) first.")

    pa: Optional[PL.PartAnalysis] = ss.analysis

    if not ss.files and pa is None:
        st.markdown(
            '<div class="empty"><h2>Ask a part anything its files can answer</h2>'
            '<p>The app reads the Excel manufacturing record, the PDF drawing and the IGES 3D model of a part, '
            'cross-checks them, and answers questions with a citation to the exact cell, drawing text or model '
            'entity behind every claim. When the files do not say, it says so.</p><ol>'
            '<li>Add the files in the sidebar (upload, a folder, or the sample dataset).</li>'
            '<li>Type the part number, for example 2280, and press <b>Analyze part</b>.</li>'
            '<li>Ask questions, or open the cross-check tab to see where the files disagree.</li></ol></div>',
            unsafe_allow_html=True)
        st.write("")
        if st.button("Try it with the sample dataset (part 2280)", type="primary"):
            sets = PL.sample_sets(SAMPLES_DIR)
            if not sets:
                sys.path.insert(0, os.path.join(ROOT, "scripts"))
                import make_samples  # noqa: WPS433
                make_samples.main(SAMPLES_DIR)
                sets = PL.sample_sets(SAMPLES_DIR)
            set_items(PL.load_sample_set(SAMPLES_DIR, list(sets)[0]), f"sample set: {list(sets)[0]}")
            ss._next_part = "2280"
            st.rerun()
        return

    if pa is None:
        kp = known_parts()
        st.markdown(f'<div class="empty"><h2>{len(ss.files)} file(s) ready</h2><p>Type a part number in the sidebar '
                    f'and press <b>Analyze part</b>.' + (" These part numbers appear in the files:" if kp else "") +
                    '</p></div>', unsafe_allow_html=True)
        if kp:
            cols = st.columns(min(6, len(kp)))
            for i, p in enumerate(kp[:12]):
                if cols[i % len(cols)].button(p, key=f"kp_{p}"):
                    ss._next_part = p
                    st.rerun()
        return

    if not pa.found:
        st.markdown(f'<div class="nf"><div class="big">Part {ui_theme.esc(pa.part_number)} is not in these files</div>'
                    f'<p>None of the {len(ss.files)} provided files contains this part number, so nothing was sent to '
                    f'the LLM. Any question about it gets: <i>{qa.NOT_FOUND}</i></p></div>', unsafe_allow_html=True)
        if pa.near_matches:
            st.markdown("**Similar part numbers found (not used automatically):**")
            for i, n in enumerate(pa.near_matches[:6]):
                if st.button(f"Analyze {n['value']} instead", key=f"near_{i}"):
                    ss._next_part = n["value"]
                    st.rerun()
                st.caption(f"{n.get('file', '')}, {n['location']}")
        kp = known_parts()
        if kp:
            st.markdown("**Part numbers that do appear in the files:**")
            cols = st.columns(min(6, len(kp)))
            for i, p in enumerate(kp[:12]):
                if cols[i % len(cols)].button(p, key=f"kpn_{p}"):
                    ss._next_part = p
                    st.rerun()
        st.markdown(ui_theme.source_cards(pa.status), unsafe_allow_html=True)
        q = st.chat_input(f"Ask about part {pa.part_number}…")
        if q:
            answer(q, pa, llm, settings)
        return

    # ---------------- hero
    n_files = sum(len(s.files) for s in pa.status.values() if s.state == "loaded")
    brain = (f"Answers by {llm.providers[0].label}, with fallback to the next provider." if llm else
             "No LLM configured: answers come from deterministic templates built from the facts.")
    load_llm = f" {pa.llm_calls_at_load} LLM call(s) while loading." if pa.llm_calls_at_load else ""
    st.markdown(
        f'<div class="hero"><div><div class="pn"><small>Part number</small>{ui_theme.esc(pa.part_number)}</div>'
        f'<div class="desc">{ui_theme.esc(describe(pa))}</div>'
        f'<div class="meta">{len(pa.facts)} facts from {n_files} file(s), analyzed in {pa.seconds:.2f} s.{load_llm} {ui_theme.esc(brain)}</div></div>'
        f'<div class="stamp{"" if pa.level == 3 else " dim"}"><div class="lv">Level {pa.level}</div>'
        f'<div class="sub">{ui_theme.esc(stamp_sub(pa))}</div></div></div>',
        unsafe_allow_html=True)
    st.markdown(ui_theme.source_cards(pa.status), unsafe_allow_html=True)
    if pa.warnings or pa.llm_notes:
        with st.expander(f"Notes from reading the files ({len(pa.warnings) + len(pa.llm_notes)})"):
            for w in pa.llm_notes + pa.warnings:
                st.markdown(f"- {ui_theme.esc(w)}")

    known_ids = [f.id for f in pa.facts]
    tab_ask, tab_check, tab_ev, tab_3d = st.tabs(["Ask", "Cross-check", "Evidence", "3D model"])

    with tab_ask:
        left, right = st.columns([3, 2], gap="large")
        with left:
            cols = st.columns(3)
            clicked = None
            for i, (label, question) in enumerate(SUGGESTED):
                if cols[i % 3].button(label, key=f"sugg_{i}", **STRETCH, help=question):
                    clicked = question
            history = st.container()
            q = st.chat_input(f"Ask about part {pa.part_number} — material, dimensions, holes, operations, agreement…")
            with history:
                for turn in ss.chats.get(pa.part_number, []):
                    with st.chat_message("user", avatar=AVATAR_USER):
                        st.write(turn["q"])
                    with st.chat_message("assistant", avatar=AVATAR_BOT):
                        render_answer(turn["ans"], pa)
                ask = q or clicked
                if ask:
                    new_pa = answer(ask, pa, llm, settings)
                    if new_pa is not pa:
                        st.rerun()
        with right:
            mism = [c for c in pa.checks if c.status == "MISMATCH"]
            ok = [c for c in pa.checks if c.status == "MATCH"]
            st.markdown(f'<div class="summary">Cross-check: <b class="{"bad" if mism else "ok"}">{len(mism)} mismatch'
                        f'{"es" if len(mism) != 1 else ""}</b>, {len(ok)} matches, {len(pa.checks) - len(mism) - len(ok)} '
                        f'other</div>', unsafe_allow_html=True)
            order = {"MISMATCH": 0, "MATCH": 2, "INFO": 3, "NOT COMPARABLE": 1}
            st.markdown(ui_theme.check_rows(sorted(pa.checks, key=lambda c: order.get(c.status, 4)), known_ids),
                        unsafe_allow_html=True)

    with tab_check:
        st.markdown(f'<div class="summary">All checks run in code after converting every value to mm. '
                    f'Tolerance: {settings.tol_rel * 100:.1f}% or {settings.tol_abs_mm:g} mm, whichever is larger.</div>',
                    unsafe_allow_html=True)
        st.markdown(ui_theme.check_rows(pa.checks, known_ids), unsafe_allow_html=True)
        missing = [f for f in pa.facts if f.source_type == "missing"]
        if missing:
            st.markdown("#### Missing information")
            st.markdown(ui_theme.chips([f.id for f in missing], pa.by_id()), unsafe_allow_html=True)
        st.caption("MATCH: the sources agree within tolerance. MISMATCH: they disagree (likely causes are explained "
                   "in the row). NOT COMPARABLE: fewer than two sources state the value. Stock size is checked as "
                   "'stock ≥ finished part', because raw stock is always larger than the machined part.")

    with tab_ev:
        rows = [f.to_row() for f in pa.facts]
        df = pd.DataFrame(rows)
        c1, c2 = st.columns([2, 3])
        srcs = c1.multiselect("Sources", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
        text = c2.text_input("Search facts", placeholder="e.g. material, Ø8.5, cycle")
        view = df[df["source"].isin(srcs)]
        if text:
            mask = view.apply(lambda r: text.lower() in " ".join(str(v) for v in r.values).lower(), axis=1)
            view = view[mask]
        st.dataframe(view, hide_index=True, **STRETCH, height=520)
        st.download_button("Download facts as CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name=f"facts_{pa.part_number}.csv", mime="text/csv")

    with tab_3d:
        ig = pa.iges
        if ig is None:
            st.markdown('<div class="empty"><h2>No IGES model for this part</h2><p>Upload the part\'s .igs/.iges '
                        'file to add geometry facts and the three-way cross-check.</p></div>', unsafe_allow_html=True)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Units in file", ig.units or "unknown")
            m2.metric("Entities", ig.total_entities)
            ext = ig.extents_mm
            m3.metric("Bounding box (mm)", " × ".join(fmt_num(v, 2) for v in ext) if ext else "n/a")
            m4.metric("Skipped entities", sum(ig.skipped.values()))
            st.caption(f"Model type: {ig.model_type}. Box {ig.bbox_kind}. {ig.unit_note}.")
            a, b = st.columns([1, 1])
            with a:
                from src.iges_parser import ENTITY_NAMES
                cnt = pd.DataFrame([{"type": t, "entity": ENTITY_NAMES.get(t, "?"), "count": n}
                                    for t, n in sorted(ig.entity_counts.items(), key=lambda kv: -kv[1])])
                st.dataframe(cnt, hide_index=True, **STRETCH, height=300)
            with b:
                if ig.holes:
                    st.dataframe(pd.DataFrame(ig.holes).rename(columns={"diameter_mm": "Ø (mm)", "count": "features",
                                                                        "circles": "circle entities"}),
                                 hide_index=True, **STRETCH)
                    st.caption("Candidate hole features: full circles grouped by axis. Outer diameters of round parts "
                               "show up here too.")
                if ig.skipped:
                    st.dataframe(pd.DataFrame([{"reason": k, "count": v} for k, v in ig.skipped.items()]),
                                 hide_index=True, **STRETCH)
            if ig.preview:
                try:
                    import plotly.graph_objects as go
                    xs, ys, zs = [], [], []
                    for poly in ig.preview[:4000]:
                        for p in poly:
                            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                        xs.append(None); ys.append(None); zs.append(None)
                    fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#14315C", width=3),
                                                 hoverinfo="skip"))
                    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                      scene=dict(aspectmode="data", xaxis_title="X mm", yaxis_title="Y mm", zaxis_title="Z mm",
                                                 bgcolor="#FAF9F3"))
                    st.plotly_chart(fig, **STRETCH)
                    st.caption("Wireframe of the edges the parser read (lines, arcs, B-spline control polygons), in mm.")
                except Exception as exc:
                    st.caption(f"Preview unavailable: {exc}")

    with st.expander("Debug log"):
        st.write({"files": ss.files_label, "status": {k: vars(v) for k, v in pa.status.items()}})
        if ss.llm_log:
            st.dataframe(pd.DataFrame(ss.llm_log), hide_index=True, **STRETCH)
        for d in (pa.debug + ss.debug)[-10:]:
            st.code(d)


try:
    main()
except Exception as exc:  # last line of defence: friendly message, details in the debug log
    ss.debug.append(traceback.format_exc())
    st.error(f"Something went wrong while drawing the page: {exc}. The details are in the debug log; "
             f"press Reset session if it persists.")
    with st.expander("Debug log"):
        st.code(ss.debug[-1])
