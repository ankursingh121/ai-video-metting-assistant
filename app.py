import streamlit as st
import time
import os
import json
import traceback
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Signal Room — AI Meeting Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Glassmorphism & SaaS Design System ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #07080d;
    --surface: rgba(17, 19, 31, 0.75);
    --surface-hover: rgba(26, 29, 46, 0.85);
    --surface-solid: #11131f;
    --border: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(124, 58, 237, 0.35);
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --teal: #06b6d4;
    --teal-glow: #22d3ee;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --text-sub: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(124, 58, 237, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 70%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
                var(--bg) !important;
}

/* Subtle background grid */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(11, 13, 22, 0.92) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, .syne-font {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* ── Brand Logo Header ── */
.brand-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}

.brand-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
}

.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #ffffff 30%, var(--accent-glow) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-sub {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Hero Title in Main View ── */
.hero-title-main {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 800;
    line-height: 1.08;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 20%, #cbd5e1 50%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub-main {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    margin-bottom: 1.5rem;
}

/* ── Glass Cards ── */
.glass-card {
    background: var(--surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.4);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
}

.glass-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}

.card-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Action Item Cards ── */
.action-item-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s ease;
}

.action-item-card:hover {
    background: rgba(255, 255, 255, 0.05);
    transform: translateX(3px);
}

/* ── Badges ── */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-purple {
    background: rgba(124, 58, 237, 0.18);
    color: #c4b5fd;
    border: 1px solid rgba(124, 58, 237, 0.35);
}

.badge-teal {
    background: rgba(6, 182, 212, 0.15);
    color: #67e8f9;
    border: 1px solid rgba(6, 182, 212, 0.3);
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* ── Pipeline Stepper ── */
.stepper-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.85rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    font-weight: 500;
    transition: all 0.2s;
}

.stepper-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active {
    background: var(--teal);
    box-shadow: 0 0 10px var(--teal);
    animation: pulse 1.2s infinite;
}

.dot-done {
    background: var(--emerald);
}

.dot-pending {
    background: var(--text-sub);
    opacity: 0.4;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.2); }
}

/* ── Input & Form Polish ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stFileUploader > div {
    background: rgba(18, 20, 32, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.25) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
}

/* Secondary Button */
.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: var(--text) !important;
}

/* ── Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(18, 20, 32, 0.6);
    padding: 0.35rem 0.5rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    font-size: 0.85rem !important;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: var(--surface-hover) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

/* ── Chat Display ── */
.chat-bubble-user {
    background: rgba(124, 58, 237, 0.18);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 12px 12px 2px 12px;
    padding: 0.85rem 1.1rem;
    margin-left: auto;
    max-width: 80%;
    margin-bottom: 0.75rem;
    font-size: 0.88rem;
    line-height: 1.6;
}

.chat-bubble-bot {
    background: rgba(18, 22, 36, 0.85);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 12px 12px 12px 2px;
    padding: 0.95rem 1.2rem;
    margin-right: auto;
    max-width: 85%;
    margin-bottom: 0.85rem;
    font-size: 0.88rem;
    line-height: 1.65;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ───────────────────────────────────────────────
defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "media_source": None,
    "media_type": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_step(key: str, state: str):
    st.session_state.pipeline_steps[key] = state

def get_dot_class(key: str) -> str:
    s = st.session_state.pipeline_steps.get(key, "pending")
    if s == "active": return "dot-active"
    if s == "done":   return "dot-done"
    return "dot-pending"

# ─── Sidebar Navigation & Controls ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-icon">⚡</div>
        <div>
            <div class="brand-name">SIGNAL ROOM</div>
            <div class="brand-sub">AI Meeting Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Pill
    st.markdown("""
    <div style="display:flex; gap:0.4rem; margin-bottom:1.25rem;">
        <span class="badge-pill badge-green">● Models Live</span>
        <span class="badge-pill badge-purple">Mistral Small</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-label">Input Source</div>', unsafe_allow_html=True)
    input_type = st.radio("Source Mode", ["YouTube Link", "Upload File"], horizontal=True, label_visibility="collapsed")

    source_url = ""
    uploaded_file = None

    if input_type == "YouTube Link":
        source_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    else:
        uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp4", "mp3", "wav", "m4a", "webm"], label_visibility="collapsed")

    col_lang, col_btn = st.columns([1, 1])
    with col_lang:
        language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡ Analyse Meeting", use_container_width=True)

    # Sample Demo Trigger
    if st.button("🧪 Try Sample Demo", kind="secondary", use_container_width=True):
        st.session_state.media_source = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        st.session_state.media_type = "youtube"
        st.session_state.result = {
            "title": "Quarterly Product & Engineering Sync",
            "transcript": (
                "Maya Chen: Good morning everyone. Today we need to align on our Q3 ingestion pipeline milestone.\n"
                "Arjun Rao: I have completed the Whisper and Sarvam integration. The accuracy on Hinglish audio is 94%.\n"
                "Maya Chen: That is fantastic. Let's make sure the Docker deployment on Render is finalized by Thursday.\n"
                "Arjun Rao: Agreed. I will also wire the ChromaDB vector retriever to ensure low-latency RAG responses.\n"
                "Maya Chen: What is our fallback plan if Sarvam API experiences rate limits?\n"
                "Arjun Rao: We fall back to OpenAI Whisper local inference automatically."
            ),
            "summary": (
                "• **Key Milestone:** The engineering team successfully integrated Whisper and Sarvam AI for bilingual speech-to-text with 94% accuracy.\n"
                "• **Infrastructure:** Render Docker deployment scheduled for completion by Thursday.\n"
                "• **Resilience:** ChromaDB vector retriever optimized; automatic fallback to local Whisper implemented to mitigate API rate limits."
            ),
            "action_items": (
                "1. **Finalize Render Docker deployment** — Owner: Arjun Rao — Deadline: Thursday\n"
                "2. **Benchmark ChromaDB vector latency** — Owner: Arjun Rao — Deadline: Next Sprint\n"
                "3. **Verify customer handoff deck** — Owner: Maya Chen — Deadline: Friday"
            ),
            "key_decisions": (
                "1. **Bilingual Engine:** Use Sarvam AI for Hinglish, local Whisper for English.\n"
                "2. **Fault-tolerance:** Local Whisper serves as fallback if cloud endpoints exceed quota."
            ),
            "open_questions": (
                "1. Will the free cloud tier provide sufficient RAM for 20-minute audio chunks?\n"
                "2. Which enterprise customers will participate in the private beta first?"
            ),
            "rag_chain": None,
        }
        st.session_state.pipeline_done = True
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I have analyzed the **Quarterly Product & Engineering Sync**. Ask me anything about the decisions, owners, or deadlines!"}
        ]
        st.rerun()

    # Pipeline Status Stepper in Sidebar
    st.markdown("---")
    st.markdown('<div class="card-label">Pipeline Progress</div>', unsafe_allow_html=True)
    steps = [
        ("audio",      "🔊", "Audio Acquisition"),
        ("transcript", "📝", "Transcription (Whisper)"),
        ("title",      "🏷️", "Title Synthesis"),
        ("summary",    "📋", "Map-Reduce Summary"),
        ("extract",    "🔍", "Action Item Extraction"),
        ("rag",        "🧠", "Vector Store & RAG"),
    ]
    for key, icon, label in steps:
        dot_css = get_dot_class(key)
        st.markdown(f"""
        <div class="stepper-row">
            <div class="stepper-dot {dot_css}"></div>
            <span>{icon} {label}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Main Content View ──────────────────────────────────────────────────────────
st.markdown("""
<div>
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
        <span class="badge-pill badge-teal">⚡ AI VIDEO ASSISTANT</span>
        <span class="badge-pill badge-green">SYSTEM READY</span>
    </div>
    <div class="hero-title-main">Your meetings, with the signal intact.</div>
    <div class="hero-sub-main">Transform raw audio and video into searchable decisions, actionable owners, and contextual RAG chat.</div>
</div>
""", unsafe_allow_html=True)

# ─── Execute Pipeline ───────────────────────────────────────────────────────────
if run_btn:
    source_to_process = None
    media_t = None

    if input_type == "YouTube Link":
        if not source_url.strip():
            st.error("⚠️ Please paste a valid YouTube URL.")
        else:
            source_to_process = source_url.strip()
            media_t = "youtube"
    else:
        if not uploaded_file:
            st.error("⚠️ Please select an audio or video file to upload.")
        else:
            os.makedirs("downloades", exist_ok=True)
            saved_path = os.path.join("downloades", uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_to_process = saved_path
            media_t = "local"

    if source_to_process:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.media_source = source_to_process
        st.session_state.media_type = media_t

        prog_banner = st.empty()

        try:
            with prog_banner.container():
                st.info("⚡ Processing pipeline initiated. Watch sidebar for live stage progress…")

            # 1. Audio Processing
            update_step("audio", "active")
            chunks = process_input(source_to_process)
            update_step("audio", "done")

            # 2. Transcription
            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            # 3. Title Generation
            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            # 4. Summarization
            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            # 5. Extraction
            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done")

            # 6. RAG Engine
            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.session_state.chat_history = [
                {"role": "assistant", "content": f"I've completed analyzing **{title}**! Ask me anything about specific discussions or owners."}
            ]
            prog_banner.success("✅ Meeting analysis completed successfully!")
            time.sleep(0.4)
            prog_banner.empty()
            st.rerun()

        except Exception as e:
            for k, _, _ in steps:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            st.error(f"❌ Pipeline encountered an error: {e}")
            with st.expander("Detailed Error Stacktrace"):
                st.code(traceback.format_exc())

# ─── Render Results Dashboard ───────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Media Player Header
    col_meta, col_media = st.columns([3, 2], gap="large")

    with col_meta:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid var(--teal);">
            <div class="card-label">📌 ANALYZED SESSION</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#ffffff; margin:0.3rem 0;">
                {r['title']}
            </div>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.75rem;">
                <span class="badge-pill badge-teal">⚡ Complete</span>
                <span class="badge-pill badge-purple">{len(r['transcript'].split())} Words</span>
                <span class="badge-pill badge-amber">RAG Indexed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_media:
        if st.session_state.media_type == "youtube":
            st.video(st.session_state.media_source)
        elif st.session_state.media_type == "local" and os.path.exists(st.session_state.media_source):
            st.audio(st.session_state.media_source)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:1.5rem;">
                <div style="font-size:1.8rem; margin-bottom:0.25rem;">🎬</div>
                <div style="font-size:0.8rem; color:var(--text-muted);">Media Stream Loaded</div>
            </div>
            """, unsafe_allow_html=True)

    # Export & Download Toolbar
    st.markdown("---")
    col_dl1, col_dl2, col_reset = st.columns([1, 1, 3])

    # Markdown Export
    md_content = f"""# {r['title']}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{r['summary']}

## Action Items
{r['action_items']}

## Key Decisions
{r['key_decisions']}

## Open Questions
{r['open_questions']}

## Full Transcript
{r['transcript']}
"""
    with col_dl1:
        st.download_button(
            "📥 Download Minutes (.md)",
            data=md_content,
            file_name=f"meeting_minutes_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            "📄 Export Plain Text (.txt)",
            data=md_content,
            file_name=f"meeting_notes_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_reset:
        if st.button("🔄 Analyze New Meeting", kind="secondary"):
            st.session_state.result = None
            st.session_state.pipeline_done = False
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.rerun()

    # ─── Multi-Tab Intelligence ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Executive Summary",
        "✅ Action Items",
        "🔑 Key Decisions",
        "❓ Open Questions",
        "📝 Full Transcript"
    ])

    with tab1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-card-header">
                <span class="card-label">EXECUTIVE BRIEF</span>
                <span class="badge-pill badge-teal">Mistral Synthesis</span>
            </div>
            <div style="font-size:0.95rem; line-height:1.8; color:var(--text);">
                {r['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="glass-card-header" style="margin-bottom:1rem;">
            <span class="card-label">ACCOUNTABILITY MATRIX</span>
            <span class="badge-pill badge-green">Extracted Tasks</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Render itemized lines
        for line in r['action_items'].split("\n"):
            clean = line.strip()
            if clean:
                st.markdown(f"""
                <div class="action-item-card">
                    <span style="font-size:0.9rem; font-weight:500;">{clean}</span>
                    <span class="badge-pill badge-teal">Assigned</span>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="glass-card-header" style="margin-bottom:1rem;">
            <span class="card-label">STRATEGIC RESOLUTIONS</span>
            <span class="badge-pill badge-purple">Decisions</span>
        </div>
        """, unsafe_allow_html=True)
        
        for line in r['key_decisions'].split("\n"):
            clean = line.strip()
            if clean:
                st.markdown(f"""
                <div class="action-item-card" style="border-left-color: var(--accent);">
                    <span style="font-size:0.9rem; font-weight:500;">{clean}</span>
                    <span class="badge-pill badge-purple">Confirmed</span>
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="glass-card-header" style="margin-bottom:1rem;">
            <span class="card-label">UNRESOLVED TOPICS</span>
            <span class="badge-pill badge-amber">Follow-ups</span>
        </div>
        """, unsafe_allow_html=True)
        
        for line in r['open_questions'].split("\n"):
            clean = line.strip()
            if clean:
                st.markdown(f"""
                <div class="action-item-card" style="border-left-color: var(--amber);">
                    <span style="font-size:0.9rem; font-weight:500;">{clean}</span>
                    <span class="badge-pill badge-amber">Needs Review</span>
                </div>
                """, unsafe_allow_html=True)

    with tab5:
        st.markdown("""
        <div class="glass-card-header">
            <span class="card-label">SPEECH-TO-TEXT RAW STREAM</span>
            <span class="badge-pill badge-teal">Whisper Engine</span>
        </div>
        """, unsafe_allow_html=True)
        st.text_area("Transcript", value=r['transcript'], height=320, label_visibility="collapsed")

    # ─── Interactive RAG Chat ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem;">
        <span style="font-size:1.4rem;">💬</span>
        <span style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;">Query the Meeting Archive</span>
    </div>
    """, unsafe_allow_html=True)

    # Quick Question Chips
    st.markdown('<div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.5rem;">SUGGESTED QUESTIONS:</div>', unsafe_allow_html=True)
    chip1, chip2, chip3, chip4 = st.columns(4)
    preset_q = None

    with chip1:
        if st.button("📌 What were the key decisions?", use_container_width=True):
            preset_q = "What were the key decisions made during this meeting?"
    with chip2:
        if st.button("👥 List task owners & deadlines", use_container_width=True):
            preset_q = "List all task owners and their respective deadlines."
    with chip3:
        if st.button("❓ What questions are unresolved?", use_container_width=True):
            preset_q = "What open questions require further follow-up?"
    with chip4:
        if st.button("⚡ Summarize in 3 bullet points", use_container_width=True):
            preset_q = "Give me a quick 3-bullet summary of the entire session."

    # Chat Bubbles
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-bubble-user">
                    <div style="font-size:0.7rem; font-weight:700; color:var(--accent-glow); margin-bottom:0.25rem;">YOU</div>
                    <div>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble-bot">
                    <div style="font-size:0.7rem; font-weight:700; color:var(--teal); margin-bottom:0.25rem;">🤖 ASSISTANT</div>
                    <div>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Chat Input Bar
    chat_col_in, chat_col_send = st.columns([5, 1], gap="small")
    with chat_col_in:
        user_query = st.text_input("Ask a question", placeholder="Ask anything about the meeting...", label_visibility="collapsed")
    with chat_col_send:
        send_clicked = st.button("Send ➔", use_container_width=True)

    query_to_send = preset_q if preset_q else (user_query if send_clicked and user_query.strip() else None)

    if query_to_send:
        st.session_state.chat_history.append({"role": "user", "content": query_to_send})
        with st.spinner("Consulting meeting transcript vector store…"):
            if r["rag_chain"]:
                ans = ask_question(r["rag_chain"], query_to_send)
            else:
                ans = "This is a demo session. In live runs, the LCEL RAG chain uses ChromaDB and Mistral AI to answer contextually!"
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()

else:
    # ── Empty State / Interactive Welcome ──
    st.markdown("""
    <div style="margin: 2.5rem 0; padding: 3rem 2rem; border-radius: 16px; background: rgba(18, 22, 36, 0.4); border: 1px solid rgba(255,255,255,0.06); text-align: center;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem;">⚡</div>
        <div style="font-family:'Syne',sans-serif; font-size: 1.8rem; font-weight:800; color:#ffffff; margin-bottom: 0.5rem;">
            Ready to Extract Signal
        </div>
        <div style="color:var(--text-muted); font-size: 0.95rem; max-width: 520px; margin: 0 auto 2rem auto; line-height: 1.6;">
            Paste a YouTube link or upload an audio/video meeting file from the sidebar. You can also click <strong>🧪 Try Sample Demo</strong> to preview the full interface immediately.
        </div>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
            <span class="badge-pill badge-purple">Whisper Speech-to-Text</span>
            <span class="badge-pill badge-teal">Mistral Summaries</span>
            <span class="badge-pill badge-green">ChromaDB RAG Search</span>
            <span class="badge-pill badge-amber">One-Click Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4-stage pipeline explanation cards
    st.markdown('<div class="card-label" style="margin-bottom:1rem;">THE INTELLIGENCE PIPELINE</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <span class="badge-pill badge-teal" style="margin-bottom:0.75rem;">Stage 01</span>
            <div style="font-weight:700; font-size:1rem; margin-bottom:0.35rem;">Audio In</div>
            <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.5;">Ingest YouTube links or local MP4/WAV files automatically chunked into 10m segments.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <span class="badge-pill badge-purple" style="margin-bottom:0.75rem;">Stage 02</span>
            <div style="font-weight:700; font-size:1rem; margin-bottom:0.35rem;">Transcribe</div>
            <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.5;">Local Whisper engine for English, or Sarvam AI for Hindi/Hinglish speech-to-text.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <span class="badge-pill badge-amber" style="margin-bottom:0.75rem;">Stage 03</span>
            <div style="font-weight:700; font-size:1rem; margin-bottom:0.35rem;">Extract Signal</div>
            <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.5;">Mistral AI extracts executive summaries, action items with owners, and open questions.</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="glass-card">
            <span class="badge-pill badge-green" style="margin-bottom:0.75rem;">Stage 04</span>
            <div style="font-weight:700; font-size:1rem; margin-bottom:0.35rem;">RAG Memory</div>
            <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.5;">Embed transcript into ChromaDB vector store and converse with your meeting contextually.</div>
        </div>
        """, unsafe_allow_html=True)