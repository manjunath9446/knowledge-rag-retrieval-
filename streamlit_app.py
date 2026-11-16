import config  # ← loads .env before any module reads API keys

import streamlit as st
import os
import sys
import json
import time
import asyncio
import concurrent.futures
from pathlib import Path

from src.pipeline import EngineeringRAG
from agent.conversational import ConversationalAgent
from access.rules import UserContext

st.set_page_config(
    page_title="Hycosys Engineering RAG",
    page_icon="🛠",
    layout="wide",
)

# ─── Project paths (used by the ingestion panel) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_UPLOAD_DIR = PROJECT_ROOT / "pdfs"
PARSER_DIR = PROJECT_ROOT / "parser"
PDF_UPLOAD_DIR.mkdir(exist_ok=True)

@st.cache_data(show_spinner=False)
def _discover_teams() -> frozenset[str]:
    chunks_path = Path("data/parsed_chunks.json")
    default_teams = frozenset(["structures"])  # Failsafe for the presentation
    
    if not chunks_path.exists():
        return default_teams
    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        teams = frozenset(c.get("team") for c in chunks if c.get("team"))
        return teams if teams else default_teams
    except Exception:
        return default_teams

AVAILABLE_TEAMS = _discover_teams()

USERS_DB = {
    "engineer@hycosys.com": UserContext("u1", "engineering", AVAILABLE_TEAMS),
    "founder@hycosys.com":  UserContext("u2", "leadership",  AVAILABLE_TEAMS),
}

@st.cache_resource(show_spinner=False)
def _build_system():
    rag = EngineeringRAG()
    agent = ConversationalAgent(rag)
    return rag, agent

# ─── Ingestion helpers ────────────────────────────────────────────────────────
def _run_async(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

def _do_ingest(uploaded_files) -> None:
    saved = []
    for uf in uploaded_files:
        dest = PDF_UPLOAD_DIR / uf.name
        if dest.exists():
            continue
        dest.write_bytes(uf.getbuffer())
        saved.append(dest)

    if not saved:
        st.info("Nothing new to ingest.")
        return

    if str(PARSER_DIR) not in sys.path:
        sys.path.insert(0, str(PARSER_DIR))

    try:
        import ingest_pdfs  # noqa: E402
    except Exception as e:
        st.error(f"Cannot load ingestion module: {type(e).__name__}: {e}")
        return

    placeholder = st.empty()
    placeholder.info(f"⏳ Ingesting {len(saved)} PDF(s) — this may take a few minutes…")
    try:
        n, total = _run_async(ingest_pdfs.ingest_files(saved))
        placeholder.success(f"✅ Ingested {n} PDF(s). Total chunks: **{total}**")
    except Exception as e:
        placeholder.error(f"❌ Ingestion failed: {type(e).__name__}: {e}")
        return

    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.rerun()

def _render_ingest_panel() -> None:
    with st.sidebar.expander("📤 Ingest new PDFs", expanded=False):
        st.caption("Upload PDFs to add them to the knowledge base.")
        uploaded = st.file_uploader(
            "Choose PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state.get('uploader_key', 0)}",
        )

        if uploaded:
            existing = {p.name for p in PDF_UPLOAD_DIR.glob("*.pdf")}
            new_files = [f for f in uploaded if f.name not in existing]

            if not new_files:
                st.info("All selected PDFs are already ingested.")
                return

            st.write(f"**{len(new_files)} new file(s):**")
            for f in new_files:
                st.write(f"- `{f.name}`")

            if st.button("🚀 Start ingestion", type="primary", use_container_width=True, key="ingest_btn"):
                _do_ingest(new_files)

# ─── Login ────────────────────────────────────────────────────────────────────
def _render_login():
    st.sidebar.title("🔐 Hycosys SSO")
    st.sidebar.caption("Sign in to access the engineering knowledge base.")

    email = st.sidebar.text_input("Corporate Email", placeholder="engineer@hycosys.com", key="login_email")
    if st.sidebar.button("Login", type="primary", use_container_width=True, key="login_btn"):
        u = USERS_DB.get(email.strip().lower())
        if u:
            st.session_state.user = u
            st.session_state.user_email = email.strip().lower()
            st.session_state.messages = []
            st.rerun()
        else:
            st.sidebar.error("Unknown email. See demo accounts below.")

    st.title("🛠 Hycosys Engineering Knowledge")
    st.markdown("### Welcome — please log in using the sidebar.")
    teams_str = ", ".join(sorted(AVAILABLE_TEAMS)) or "(none ingested)"
    st.info(f"**Available teams in corpus:** {teams_str}")

# ─── Auth gate ───────────────────────────────────────────────────────────────
if "user" not in st.session_state or st.session_state.user is None:
    _render_login()
    st.stop()

if st.session_state.get("user_email") in USERS_DB:
    st.session_state.user = USERS_DB[st.session_state.user_email]

user: UserContext = st.session_state.user
user_email: str = st.session_state.get("user_email", user.user_id)

# ─── Sidebar (logged-in state) ───────────────────────────────────────────────
st.sidebar.title("🔐 Hycosys SSO")
st.sidebar.success(f"✅ {user_email}")
st.sidebar.caption(f"Clearance: **{user.clearance}**")
if user.teams:
    st.sidebar.caption(f"Teams: {', '.join(sorted(user.teams))}")
else:
    st.sidebar.caption("Teams: *(none)*")

if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
    st.session_state.user = None
    st.session_state.user_email = None
    st.session_state.messages = []
    st.rerun()

rag, agent = _build_system()

st.sidebar.divider()
st.sidebar.markdown("### 📚 System status")
st.sidebar.caption(f"Chunks indexed: **{len(rag.chunks)}**")
if rag.db_conn:
    st.sidebar.caption(f"SQL tables: **{len(rag.table_schemas)}**")
else:
    st.sidebar.caption("SQL tables: *not loaded*")
st.sidebar.caption(f"Corpus teams: {', '.join(sorted(AVAILABLE_TEAMS)) or '—'}")

st.sidebar.divider()
_render_ingest_panel()

if st.session_state.get("messages"):
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        if hasattr(agent, "reset"):
            try:
                agent.reset()
            except Exception:
                pass
        st.rerun()

# ─── Assistant rendering ────────────────────────────────────────────────────
def _route_badge(sources: list, refused: bool) -> tuple[str, str]:
    if refused:
        return "REFUSED", "#d7263d"
    if any(s.get("doc_id") == "SQL_DB" for s in sources):
        return "SQL", "#1f6feb"
    if any("image_path" in s for s in sources):
        return "VQA", "#8957e5"
    return "VECTOR", "#2ea043"

def _render_assistant(m: dict) -> None:
    refused = m.get("refused", False)

    if refused:
        st.error(m["content"])
        badge, color = _route_badge([], True)
        st.markdown(
            f"<span style='background:{color};color:white;padding:2px 8px;"
            f"border-radius:4px;font-size:0.75em;'>{badge}</span>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(m["content"], unsafe_allow_html=True)

    sources = m.get("sources", [])
    for s in sources:
        p = s.get("image_path")
        if p and os.path.exists(p):
            st.image(p, caption=f"Source: {s.get('title', '')}")

    if sources:
        with st.expander(f"📚 Traceable sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- **{s.get('doc_id', '?')}** — {s.get('title', '?')}")

    badge, color = _route_badge(sources, False)
    conf = m.get("confidence")
    parts = [
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.75em;'>{badge}</span>"
    ]
    if conf is not None:
        parts.append(
            f"<span style='color:gray;font-size:0.8em;margin-left:8px;'>"
            f"confidence: {conf:.2f}</span>"
        )
    st.markdown(" ".join(parts), unsafe_allow_html=True)

# ─── Main chat UI ───────────────────────────────────────────────────────────
st.title("🛠 Hycosys Engineering Knowledge")
st.caption("Ask about spindle failure, hardness, test-rig locations, Campbell diagram, etc.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"], unsafe_allow_html=True)
        else:
            _render_assistant(m)

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and analyzing..."):
            t0 = time.perf_counter()
            try:
                res = agent.ask(prompt, user)
            except Exception as e:
                res = {
                    "answer": f"⚠️ **Error:** {type(e).__name__}: {e}",
                    "refused": True,
                    "sources": [],
                    "confidence": 0.0,
                }
            latency = time.perf_counter() - t0

        msg = {
            "role": "assistant",
            "content": res["answer"],
            "refused": res.get("refused", False),
            "sources": res.get("sources", []),
            "confidence": res.get("confidence", 0.0),
            "latency": latency,
        }
        _render_assistant(msg)
        st.caption(f"⏱️ {latency:.2f}s")

    st.session_state.messages.append(msg)