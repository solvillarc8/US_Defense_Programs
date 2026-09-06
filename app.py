"""Professional Streamlit interface for The Program Desk."""

import hashlib
from html import escape

import streamlit as st

from agent import ask
from config import SOURCE_DOCUMENT_LINKS, AVAILABLE_FISCAL_YEARS
from tools import set_year_scope, warm_up
from usaspending import latest_awards
from voice import transcribe_audio


st.set_page_config(page_title="The Program Desk", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading search indexes (one-time)…")
def _warm_indexes() -> bool:
    # Runs ONCE per server process: loads the embedding model, builds each
    # corpus's BM25 index and loads the reranker, so the first real question
    # doesn't pay 15-30s of cold-start cost while the presenter waits.
    warm_up()
    return True


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_latest_awards(query: str) -> list[dict]:
    return latest_awards(query, months_back=6, limit=3)


_warm_indexes()

# Every font on the page is Arial — the "font-family" rule below is the ONE
# place that's declared, and every other selector inherits it rather than
# repeating it (the .retrieval-card block used to override this with a
# serif face; it no longer does, on purpose).
st.markdown("""
<style>
:root { --navy:#0b3558; --blue:#145f8c; --ink:#17212b; --muted:#5b6977; --line:#d8e0e7; --paper:#f5f7f9; }

/* This app has ONE deliberate visual design (light, navy/white) and does
   not offer a dark mode — without this, a viewer with OS/browser dark
   mode enabled gets Streamlit's own dark theme underneath, and its CSS
   variables (which most built-in widgets — radio labels, buttons, inputs
   — read their text/background colors FROM) win over broad selectors
   like `color:var(--ink)` on outer elements. Overriding Streamlit's own
   theme variables directly is what actually forces every widget, not
   just the ones this file writes CSS for, to use the light palette. */
:root, html, body, .stApp {
  --text-color: #17212b !important;
  --background-color: #ffffff !important;
  --secondary-background-color: #eef2f5 !important;
  --primary-color: #0b3558 !important;
  color-scheme: light !important;
}

html, body, [class*="css"], .stApp, button, input, textarea { font-family:Arial,Helvetica,sans-serif!important; color:var(--ink); }
.stApp { background:#fff; }
/* Backstop for the light theme pinned in .streamlit/config.toml: force
   all body text dark. Links are excluded so source links stay blue and
   the LIVE label stays green. */
[data-testid="stMarkdownContainer"] :not(a):not(a *),
[data-testid="stChatMessage"] :not(a):not(a *),
[data-testid="stRadio"] label span, [data-testid="stSidebar"] p, .stApp p, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th, .stApp label {
  color:var(--ink)!important;
}
.stApp a, .source-link a { color:var(--blue)!important; }
.live-label { color:#19643a!important; }
/* Button labels also live inside stMarkdownContainer — keep them readable
   on their own backgrounds (navy Send/Refresh, white Preview page). */
.stButton>button [data-testid="stMarkdownContainer"] p { color:#fff!important; }
.source-card .stButton>button [data-testid="stMarkdownContainer"] p { color:var(--blue)!important; }
.source-card .stButton>button:hover [data-testid="stMarkdownContainer"] p { color:#fff!important; }
[data-testid="stPills"] button p { color:var(--ink)!important; }
[data-testid="stPills"] button[aria-checked="true"] p, [data-testid="stPills"] button[data-selected="true"] p { color:#fff!important; }
.stApp input, .stApp textarea { color:var(--ink)!important; background:#fff!important; }
.block-container { max-width:1180px; padding-top:1.6rem; padding-bottom:7.5rem; }
[data-testid="stSidebar"] { background:#eef2f5; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top:1.6rem; }
[data-testid="stChatMessage"] { background:transparent; border:0; padding:.65rem 0; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display:none; }
.program-header { border-top:6px solid var(--navy); border-bottom:1px solid var(--line); padding:1.35rem 0 1.1rem; margin-bottom:1.5rem; }
.program-kicker { color:var(--blue); font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.45rem; }
.program-title { color:var(--navy); font-size:2.35rem; font-weight:700; margin:0; }
.program-subtitle { color:var(--muted); font-size:1rem; margin:.45rem 0 0; max-width:780px; }
.section-label { color:var(--navy); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin:1.1rem 0 .55rem; }
.source-card { border:1px solid var(--line); border-left:4px solid var(--blue); background:var(--paper); padding:.85rem 1rem; margin:.55rem 0; }
.source-title { color:var(--navy); font-weight:700; font-size:.95rem; }
.source-meta { color:var(--muted); font-size:.82rem; margin-top:.15rem; }
.source-link a { color:var(--blue); font-size:.84rem; font-weight:700; text-decoration:none; }
.source-link a:hover { text-decoration:underline; }
.retrieval-card { background:#f8f5ef; border:1px solid #ddd5c8; padding:.9rem 1rem; margin:.65rem 0; color:#33302b; font-size:.9rem; line-height:1.5; white-space:pre-wrap; overflow-wrap:anywhere; }
.sidebar-heading { color:var(--navy); font-size:1rem; font-weight:700; margin:.7rem 0; }
.sidebar-copy { color:var(--muted); font-size:.87rem; line-height:1.45; }
.live-label { color:#19643a; font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-left:.35rem; }
.sidebar-rule { border-top:1px solid #cdd6de; margin:1.2rem 0; }
.stButton>button, .stFormSubmitButton>button { background:var(--navy); color:#fff; border:1px solid var(--navy); border-radius:3px; font-weight:700; height:3rem; min-height:3rem; }
.stButton>button:hover { background:var(--blue); border-color:var(--blue); color:#fff; }
.stFormSubmitButton>button:hover { background:var(--blue); border-color:var(--blue); color:#fff; }
.stTextInput input { border-radius:3px; border-color:#aebbc6; height:3rem; min-height:3rem; }
[data-testid="stForm"] { border:0; padding:0; }
details { border:1px solid var(--line)!important; border-radius:3px!important; }
.source-card .stButton>button { background:#fff; color:var(--blue); border:1px solid var(--blue); border-radius:3px; font-weight:600; height:2rem; min-height:2rem; font-size:.8rem; padding:0 .8rem; margin-top:.5rem; }
.source-card .stButton>button:hover { background:var(--blue); color:#fff; }
.preview-panel { position:sticky; top:1rem; border:1px solid var(--line); background:var(--paper); padding:1.1rem 1.2rem; min-height:220px; }
.preview-empty { color:var(--muted); font-size:.88rem; line-height:1.5; }
.preview-title { color:var(--navy); font-weight:700; font-size:1.02rem; }
.preview-meta { color:var(--muted); font-size:.83rem; margin:.2rem 0 .8rem; }

/* --- Watchlist (sidebar) --- */
.watch-card { border:1px solid var(--line); border-left:4px solid var(--navy); background:#fff; padding:.7rem .85rem; margin:.5rem 0; }
.watch-program { color:var(--navy); font-weight:700; font-size:.88rem; }
.watch-flag { display:block; font-size:.78rem; margin-top:.3rem; }
.watch-flag.new { color:#19643a; font-weight:700; }
.watch-flag.changed { color:#8a5a12; font-weight:700; }
.watch-flag.quiet { color:var(--muted); }

header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
  background:#fff!important; border-bottom:1px solid var(--line);
}
header[data-testid="stHeader"] * { color:var(--ink)!important; }

/* Nothing on the page should ever cause horizontal scroll — a fixed-width
   child anywhere (an unwrapped long URL, a wide code block) is the usual
   cause. Belt-and-suspenders containment instead of chasing each case. */
html, body { overflow-x:hidden; }
.stApp, .block-container, [data-testid="stAppViewContainer"] { max-width:100%; overflow-x:hidden; }
img, pre, code { max-width:100%; }

/* --- Composer: sticky at the bottom of the main content column, like
   Claude/WhatsApp — always in the same place, so you scroll the
   conversation UP to read it rather than hunting for the input box.
   position:sticky (not fixed) deliberately: sticky respects its own
   parent's actual rendered width automatically, so it's correctly
   positioned whether the sidebar is expanded, resized, or collapsed —
   fixed positioning needed a guessed sidebar-width offset that broke
   under any of those, which is what caused the layout to overflow. The
   tradeoff is the composer must be the LAST thing drawn in the script
   (see the bottom of this file) — sticky only pins once you've scrolled
   past its normal position, so anything drawn after it would render
   underneath it. */
.st-key-composer {
  position:sticky; bottom:0; z-index:999;
  background:#fff; border-top:1px solid var(--line); padding:.8rem 0 1rem;
  margin-top:1rem;
}

/* Composer row alignment: pin every element in the row to the exact same
   3rem height and bottom-align the columns themselves — relying on
   vertical_alignment="bottom" alone wasn't enough once the mic recorder's
   own internal height (which grows once recording starts, showing a
   waveform + timer) stopped matching the text box and Send button. */
.st-key-composer [data-testid="stHorizontalBlock"] { align-items:flex-end!important; }
.st-key-composer [data-testid="stTextInput"] input { height:3rem; }
.st-key-composer .stButton>button { height:3rem; min-height:3rem; }
.mic-col [data-testid="stAudioInput"] {
  border:1px solid var(--navy); border-radius:3px; background:#fff;
  height:3rem; max-height:3rem; overflow:hidden;
}
</style>
""", unsafe_allow_html=True)


def _document_link(source: str, page: int | None = None) -> tuple[str, str] | None:
    for item in SOURCE_DOCUMENT_LINKS:
        if item["match"] in source:
            url = item["url"] + (f"#page={page}" if page else "")
            return item["label"], url
    return None


def _select_source(citation: dict) -> None:
    # Runs before the script reruns (a button's on_click callback), so this
    # is a safe place to update session_state ahead of the preview panel
    # being redrawn further down in the same run.
    st.session_state.selected_source = citation


def _citation_card(citation: dict, key: str) -> None:
    page_start, page_end = citation["page_start"], citation["page_end"]
    pages = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    link = _document_link(citation["source"], page_start)
    link_html = ""
    if link:
        label, url = link
        link_html = f'<div class="source-link"><a href="{escape(url)}" target="_blank" rel="noopener noreferrer">Open {escape(label)}</a></div>'
    st.markdown(f"""
    <div class="source-card">
      <div class="source-title">{escape(str(citation['program']))}</div>
      <div class="source-meta">Pages {escape(pages)} | {escape(str(citation['source']))}</div>
      {link_html}
    </div>
    """, unsafe_allow_html=True)
    if citation.get("image_paths"):
        st.button("Preview page", key=key, on_click=_select_source, args=(citation,))


def _retrievals(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander("Retrieved evidence and tool trace"):
        for call in tool_calls:
            tool_input = call.get("input", {})
            query = tool_input.get("query") or tool_input.get("program") or ""
            st.markdown(f"**{call['name']}**")
            if call["name"] == "search_usaspending_awards":
                st.markdown('<span class="live-label">Live data</span> &nbsp; [Open USAspending.gov](https://www.usaspending.gov/)', unsafe_allow_html=True)
            if query:
                st.caption(f"Query: {query}")
            st.markdown(f'<div class="retrieval-card">{escape(call["result"][:4000])}</div>', unsafe_allow_html=True)


def _evidence(citations: list[dict], tool_calls: list[dict], key_prefix: str) -> None:
    if citations:
        st.markdown('<div class="section-label">Verified sources</div>', unsafe_allow_html=True)
        for i, citation in enumerate(citations):
            _citation_card(citation, key=f"{key_prefix}-src-{i}")
    _retrievals(tool_calls)


def _preview_panel() -> None:
    """The persistent right-hand panel — clicking 'Preview page' on any
    source card (past or current turn) fills this in place, so the reader
    can check a citation against its actual source page without losing
    their spot in the conversation."""
    st.markdown('<div class="section-label">Source preview</div>', unsafe_allow_html=True)
    selected = st.session_state.get("selected_source")
    st.markdown('<div class="preview-panel">', unsafe_allow_html=True)
    if not selected:
        st.markdown(
            '<div class="preview-empty">Select "Preview page" on any '
            "source below to view the cited page here.</div>",
            unsafe_allow_html=True,
        )
    else:
        page_start, page_end = selected["page_start"], selected["page_end"]
        pages = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        st.markdown(f'<div class="preview-title">{escape(str(selected["program"]))}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="preview-meta">Pages {escape(pages)} | {escape(str(selected["source"]))}</div>',
            unsafe_allow_html=True,
        )
        for image_path in selected.get("image_paths", []):
            st.image(image_path, use_container_width=True)
        link = _document_link(selected["source"], page_start)
        if link:
            label, url = link
            st.markdown(
                f'<div class="source-link"><a href="{escape(url)}" target="_blank" '
                f'rel="noopener noreferrer">Open {escape(label)}</a></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_latest_awards() -> None:
    """Short, live: the 3 largest DoD awards in the last 6 months for one
    program keyword, straight from USAspending.gov. Cached for an hour so
    reruns don't re-hit the API."""
    program = st.text_input(
        "Program", value=st.session_state.get("latest_program", "F-35"),
        key="latest_program", label_visibility="collapsed",
        placeholder="Program keyword, e.g. F-35",
    ).strip()
    if not program:
        return
    try:
        awards = _cached_latest_awards(program)
    except Exception:
        st.markdown('<div class="sidebar-copy">Live award data unavailable right now.</div>', unsafe_allow_html=True)
        return
    if not awards:
        st.markdown(f'<div class="sidebar-copy">No DoD awards matching "{escape(program)}" in the last 6 months.</div>', unsafe_allow_html=True)
        return
    for a in awards:
        amount = f"${a['amount']:,.0f}" if isinstance(a["amount"], (int, float)) else "n/a"
        st.markdown(
            f'<div class="watch-card"><div class="watch-program">{escape(a["recipient"])}</div>'
            f'<span class="watch-flag new">{amount}</span>'
            f'<span class="watch-flag quiet">{escape(str(a["date"]))} · Award {escape(str(a["award_id"]))}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sidebar-copy"><a href="https://www.usaspending.gov/" target="_blank">USAspending.gov</a> · largest of the most recent DoD awards (6 months)</div>', unsafe_allow_html=True)


if "year_filter" not in st.session_state:
    st.session_state.year_filter = []

with st.sidebar:
    st.markdown('<div class="sidebar-heading">Filter by year</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-copy">Select one or more years. Nothing selected = latest year (FY2026). '
        'Applies to GAO and the Weapons Book; CRS covers FY2026 only.</div>',
        unsafe_allow_html=True,
    )
    st.pills(
        "Filter by year", AVAILABLE_FISCAL_YEARS, selection_mode="multi",
        key="year_filter", label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">Latest awards <span class="live-label">Live</span></div>', unsafe_allow_html=True)
    _render_latest_awards()
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">Source library</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-copy">Official public sources used by the desk.</div>', unsafe_allow_html=True)
    for item in SOURCE_DOCUMENT_LINKS:
        live = ' <span class="live-label">Live</span>' if item.get("live") else ""
        st.markdown(f"[{item['label']}]({item['url']}){live}", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">Good starting questions</div>', unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-copy">
    Compare GAO and DoD on CVN 78.<br><br>
    What changed in Sentinel's schedule?<br><br>
    What did Congress fund for selected aircraft?<br><br>
    Show recent F-35 contract awards.
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-copy">Public, unclassified sources only. Live records may describe older base awards with recent transaction activity.</div>', unsafe_allow_html=True)


st.markdown("""
<div class="program-header">
  <div class="program-kicker">US defense acquisition intelligence</div>
  <h1 class="program-title">The Program Desk</h1>
  <p class="program-subtitle">Evidence-led program briefs from GAO, the DoD Weapons Book, CRS, and bounded live contract-award data.</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "display" not in st.session_state:
    st.session_state.display = []
if "draft_question" not in st.session_state:
    st.session_state.draft_question = ""
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None


def _handle_send() -> None:
    # An on_click callback runs BEFORE the script reruns, so clearing
    # draft_question here is safe — doing it after the text box below
    # already rendered this run would raise a Streamlit widget-state error.
    text = st.session_state.draft_question.strip()
    if text:
        st.session_state.pending_question = text
    st.session_state.draft_question = ""


# `pending_question` was set by _handle_send's callback during the PREVIOUS
# run (button callbacks run before the script body), so it's already known
# here at the top — we don't need to draw the composer widgets first to
# get it. That's what lets the composer live at the very BOTTOM of the
# script (see the end of this file) with position:sticky, appearing after
# every message like a real chat UI, while still processing this turn's
# question up here where the conversation is rendered.
question = st.session_state.pop("pending_question", None)

# Year scope is ENFORCED at the tool level (tools.set_year_scope): every
# GAO / Weapons Book search is hard-filtered to the selected years no
# matter what the model does. Empty selection = latest year (FY2026).
selected_years = list(st.session_state.year_filter or [])
set_year_scope(selected_years)
scope_label = ", ".join(selected_years) if selected_years else "FY2026"
question_for_agent = (
    f"{question}\n\n(Active year filter: {scope_label}. Searches are already "
    f"restricted to these years; cite the year with each source.)"
    if question else question
)

has_conversation = bool(st.session_state.display) or question is not None
if has_conversation:
    col_main, col_preview = st.columns([3, 2], gap="large")
else:
    # No source preview panel on the empty/first-load state — it has
    # nothing to show yet and was cluttering the initial screen. It
    # appears the moment there's a real conversation (a citation to
    # preview), not before.
    col_main = st.container()
    col_preview = None

with col_main:
    for turn_i, turn in enumerate(st.session_state.display):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            _evidence(turn.get("citations", []), turn.get("tool_calls", []), key_prefix=f"turn{turn_i}")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            out = {}
            # The tool-call round (retrieval) yields no text, so without
            # this the screen is blank until the first answer token. The
            # note disappears as soon as streaming starts.
            searching = st.empty()
            searching.caption(f"Searching sources ({scope_label})…")

            def _stream():
                first = True
                for fragment in ask(question_for_agent, history=st.session_state.history[-8:], out=out):
                    if first:
                        searching.empty()
                        first = False
                    yield fragment
                searching.empty()

            try:
                full_answer = st.write_stream(_stream())
            except RuntimeError as error:
                searching.empty()
                st.error(str(error))
                st.stop()
            _evidence(out.get("citations", []), out.get("tool_calls", []), key_prefix="live")

        st.session_state.history += [
            {"role": "user", "content": question_for_agent},
            {"role": "assistant", "content": full_answer},
        ]
        st.session_state.display += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": full_answer, "tool_calls": out.get("tool_calls", []), "citations": out.get("citations", [])},
        ]

if col_preview is not None:
    with col_preview:
        _preview_panel()

# The composer: mic, text box, Send — one row, WhatsApp-style, sticky at
# the bottom (see .st-key-composer CSS). Must be the LAST thing drawn —
# sticky only pins once scrolled past its normal position, so anything
# drawn after it here would render underneath it instead of above.
with st.container(key="composer"):
    # Columns fix the VISUAL order (input, mic, send — mic on the right,
    # next to Send) independent of fill order below. Mic must still be
    # PROCESSED before the text box is drawn, since that's how its
    # transcription gets into the box (setting a widget's value has to
    # happen before that widget is instantiated) — filling col_mic first
    # in code doesn't move it visually, because its column position was
    # already fixed by st.columns() above.
    col_input, col_mic, col_send = st.columns([7, 1, 1.2], vertical_alignment="bottom")
    with col_mic:
        st.markdown('<div class="mic-col">', unsafe_allow_html=True)
        audio_value = st.audio_input("Record", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing voice input"):
                transcribed = transcribe_audio(audio_bytes)
            if transcribed:
                # Must happen BEFORE the text_input below is instantiated —
                # that's how a widget's value is set programmatically.
                st.session_state.draft_question = transcribed

    with col_input:
        st.text_input(
            "Question", key="draft_question", label_visibility="collapsed",
            placeholder="Ask about a program, budget decision, source divergence, or recent award",
            on_change=_handle_send,  # pressing Enter fires on_change -> sends, same as clicking Send
        )
    with col_send:
        st.button("Send", use_container_width=True, on_click=_handle_send)
