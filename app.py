from __future__ import annotations

import html

import streamlit as st

from ragbot.chunking import chunk_pages
from ragbot.config import Settings
from ragbot.document_loader import load_documents
from ragbot.embeddings import HashingEmbedder
from ragbot.llm import generate_answer
from ragbot.vector_store import JsonVectorStore


st.set_page_config(page_title="Document Q&A RAG Bot", page_icon="R", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --ink: #111936;
        --muted: #657090;
        --muted-2: #8c95ad;
        --line: #e3e7f2;
        --panel: #ffffff;
        --soft: #f6f8ff;
        --purple: #6757f5;
        --purple-2: #8b79ff;
        --green: #35be4b;
        --orange: #ff9f1c;
        --shadow: 0 18px 45px rgba(28, 35, 70, 0.09);
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(103, 87, 245, .12), transparent 28%),
            linear-gradient(180deg, #fbfcff 0%, #ffffff 50%, #fbfcff 100%);
    }

    .block-container {
        padding-top: 1.45rem;
        padding-bottom: 2.4rem;
        max-width: 1260px;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,.96), rgba(248,250,255,.96)),
            radial-gradient(circle at 10% 0%, rgba(103,87,245,.13), transparent 32%);
        border-right: 1px solid var(--line);
        box-shadow: 10px 0 35px rgba(32, 40, 80, .04);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--ink);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: var(--muted);
    }

    .topbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 1rem;
        color: var(--ink);
        font-weight: 650;
        margin-bottom: 1.2rem;
    }

    .top-sep {
        width: 1px;
        height: 28px;
        background: var(--line);
    }

    .avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        color: #fff;
        font-weight: 760;
        background: linear-gradient(135deg, var(--purple), var(--purple-2));
        box-shadow: 0 14px 28px rgba(103, 87, 245, .26);
    }

    .brand-card {
        display: flex;
        align-items: center;
        gap: .8rem;
        margin: .8rem 0 2.3rem;
    }

    .brand-icon, .hero-icon, .side-icon, .metric-icon {
        display: grid;
        place-items: center;
        color: #fff;
        font-weight: 800;
        box-shadow: 0 12px 28px rgba(103, 87, 245, .22);
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--purple), var(--purple-2));
    }

    .brand-title {
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 780;
        line-height: 1.05;
    }

    .brand-subtitle {
        color: var(--purple);
        font-size: .92rem;
        font-weight: 650;
        margin-top: .18rem;
    }

    .side-section-title {
        color: var(--purple);
        font-size: .82rem;
        font-weight: 820;
        text-transform: uppercase;
        letter-spacing: .12em;
        margin: 1rem 0 1.1rem;
    }

    .side-row {
        display: flex;
        align-items: center;
        gap: .9rem;
        margin: .95rem 0;
    }

    .side-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        box-shadow: none;
    }

    .side-icon.purple { background: #ece9ff; color: var(--purple); }
    .side-icon.green { background: #e7f8ed; color: #0e9f63; }
    .side-icon.orange { background: #fff1d8; color: var(--orange); }

    .side-label {
        color: var(--muted);
        font-size: .92rem;
        margin-bottom: .1rem;
    }

    .side-value {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 780;
        line-height: 1.1;
    }

    .side-value.small {
        color: #009966;
        font-size: .94rem;
        font-weight: 700;
        word-break: break-word;
    }

    .side-value.orange {
        color: var(--orange);
        font-size: .94rem;
    }

    .sidebar-rule {
        height: 1px;
        background: var(--line);
        margin: 1.65rem 0;
    }

    .hero {
        display: grid;
        grid-template-columns: 80px 1fr 170px;
        gap: 1.25rem;
        align-items: center;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 2rem 2.25rem;
        background:
            radial-gradient(circle at 92% 20%, rgba(103,87,245,.13), transparent 30%),
            linear-gradient(135deg, rgba(255,255,255,.98), rgba(247,249,255,.96));
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
    }

    .hero-icon {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--purple), var(--purple-2));
        font-size: 1.15rem;
    }

    .hero-title {
        color: var(--ink);
        font-size: 2.45rem;
        line-height: 1.08;
        font-weight: 820;
        margin-bottom: .55rem;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 760px;
    }

    .hero-art {
        justify-self: end;
        width: 132px;
        height: 132px;
        border-radius: 22px;
        background:
            linear-gradient(145deg, rgba(255,255,255,.72), rgba(230,226,255,.92));
        border: 1px solid rgba(103,87,245,.16);
        position: relative;
    }

    .hero-art:before {
        content: "";
        position: absolute;
        left: 28px;
        top: 20px;
        width: 70px;
        height: 88px;
        border-radius: 8px;
        background: #ffffff;
        border: 5px solid #8b79ff;
        box-shadow: 0 16px 28px rgba(103,87,245,.18);
    }

    .hero-art:after {
        content: "";
        position: absolute;
        right: 15px;
        bottom: 24px;
        width: 47px;
        height: 47px;
        border-radius: 50%;
        border: 8px solid #5f54d9;
        box-shadow: 28px 28px 0 -20px #5f54d9;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.1rem;
        margin: 1.25rem 0 1.35rem;
    }

    .metric-card {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.35rem 1.45rem;
        background: rgba(255,255,255,.96);
        box-shadow: 0 15px 34px rgba(28,35,70,.08);
    }

    .metric-icon {
        width: 58px;
        height: 58px;
        border-radius: 10px;
        font-size: 1.3rem;
    }

    .metric-icon.purple { background: linear-gradient(135deg, var(--purple), var(--purple-2)); }
    .metric-icon.green { background: linear-gradient(135deg, #31c74a, #51d766); }
    .metric-icon.orange { background: linear-gradient(135deg, #ff9417, #ffb23d); }

    .metric-label {
        color: var(--muted);
        font-size: .92rem;
        text-transform: uppercase;
        letter-spacing: .06em;
        font-weight: 720;
        margin-bottom: .25rem;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 850;
        line-height: 1;
    }

    .question-card, .trust-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.97);
        box-shadow: var(--shadow);
    }

    .question-card {
        padding: 1.35rem 1.45rem 1.5rem;
        margin-bottom: 1.35rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.13rem;
        font-weight: 780;
        margin-bottom: .8rem;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 1px solid #dce2ef;
        background: #f9fbff;
        min-height: 130px;
        color: var(--ink);
        font-size: 1rem;
        padding: 1rem;
    }

    div[data-testid="stTextArea"] label {
        display: none;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 780;
        border: 0;
        background: linear-gradient(135deg, var(--purple), #5545ee);
        box-shadow: 0 16px 28px rgba(103,87,245,.24);
    }

    .stButton > button:hover {
        border: 0;
        filter: brightness(1.03);
    }

    .trust-card {
        display: grid;
        grid-template-columns: 64px 1fr 130px;
        gap: 1rem;
        align-items: center;
        padding: 1.05rem 1.45rem;
        background:
            radial-gradient(circle at 95% 50%, rgba(103,87,245,.14), transparent 28%),
            linear-gradient(135deg, #ffffff, #fbfaff);
    }

    .trust-title {
        color: var(--ink);
        font-size: 1.1rem;
        font-weight: 780;
        margin-bottom: .25rem;
    }

    .trust-copy, .source-meta {
        color: var(--muted);
        font-size: .95rem;
        line-height: 1.55;
    }

    .answer-box {
        border: 1px solid #c9f0e7;
        border-left: 5px solid #0f9f86;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background: #effdfa;
        color: #142321;
        margin-bottom: 1.1rem;
        line-height: 1.58;
        box-shadow: 0 12px 28px rgba(15,159,134,.08);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(28,35,70,.05);
    }

    .stSlider [data-baseweb="slider"] > div {
        color: var(--purple);
    }

    @media (max-width: 850px) {
        .hero {
            grid-template-columns: 1fr;
        }
        .hero-art {
            display: none;
        }
        .metric-strip {
            grid-template-columns: 1fr;
        }
        .trust-card {
            grid-template-columns: 1fr;
        }
        .topbar {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_components() -> tuple[Settings, HashingEmbedder, JsonVectorStore]:
    settings = Settings()
    embedder = HashingEmbedder(settings.vector_size)
    store = JsonVectorStore(settings.persist_dir, settings.store_file)
    return settings, embedder, store


def refresh_index(settings: Settings, embedder: HashingEmbedder) -> int:
    pages = load_documents(settings.data_dir)
    chunks = chunk_pages(pages, settings.chunk_size_words, settings.chunk_overlap_words)
    store = JsonVectorStore(settings.persist_dir, settings.store_file)
    store.reset()
    store.add_chunks(chunks, embedder)
    return len(chunks)


settings, embedder, store = load_components()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-card">
          <div class="brand-icon">QA</div>
          <div>
            <div class="brand-title">RAG Bot</div>
            <div class="brand-subtitle">Document Q&A</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-section-title">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Source chunks", min_value=1, max_value=5, value=settings.top_k)
    st.caption("Lower values use less memory. Higher values provide more context.")

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section-title">Index Information</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="side-row">
          <div class="side-icon purple">C</div>
          <div>
            <div class="side-label">Indexed chunks</div>
            <div class="side-value">{len(store.items)}</div>
          </div>
        </div>
        <div class="side-row">
          <div class="side-icon green">D</div>
          <div>
            <div class="side-label">Vector store</div>
            <div class="side-value small">{settings.persist_dir / settings.store_file}</div>
          </div>
        </div>
        <div class="side-row">
          <div class="side-icon orange">M</div>
          <div>
            <div class="side-label">Ollama model</div>
            <div class="side-value orange">{settings.ollama_model}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section-title">After Adding Docs</div>', unsafe_allow_html=True)
    if st.button("Refresh Index", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            chunk_count = refresh_index(settings, embedder)
        st.cache_resource.clear()
        st.success(f"Indexed {chunk_count} chunks. Refreshing...")
        st.rerun()

if not store.items:
    st.warning("The vector store is empty. Use Refresh Index in the sidebar or run index.py --reset.")
    st.stop()

unique_sources = sorted({item["metadata"]["source"] for item in store.items})

st.markdown(
    """
    <div class="topbar">
      <div>Deploy</div>
      <div class="top-sep"></div>
      <div>Theme</div>
      <div class="avatar">P</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-icon">RAG</div>
      <div>
        <div class="hero-title">Document Q&A RAG Bot</div>
        <div class="hero-copy">
          Ask grounded questions across the indexed document collection,
          inspect retrieved chunks, and verify each answer against its cited sources.
        </div>
      </div>
      <div class="hero-art"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-strip">
      <div class="metric-card">
        <div class="metric-icon purple">F</div>
        <div>
          <div class="metric-label">Documents</div>
          <div class="metric-value">{len(unique_sources)}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon green">G</div>
        <div>
          <div class="metric-label">Indexed Chunks</div>
          <div class="metric-value">{len(store.items)}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon orange">K</div>
        <div>
          <div class="metric-label">Top-K Retrieval</div>
          <div class="metric-value">{top_k}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="question-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Ask a question</div>', unsafe_allow_html=True)
question = st.text_area(
    "Question",
    height=130,
    placeholder="Ask a question about the indexed documents...",
)
ask_clicked = st.button("Ask Documents", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="trust-card">
      <div class="side-icon purple">OK</div>
      <div>
        <div class="trust-title">Grounded. Transparent. Verifiable.</div>
        <div class="trust-copy">
          Answers are generated from your documents and include source chunks so every response can be checked.
        </div>
      </div>
      <div class="hero-art" style="width: 112px; height: 86px;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if ask_clicked and question.strip():
    with st.spinner("Retrieving relevant chunks..."):
        chunks = store.query(question.strip(), embedder, top_k)

    with st.spinner("Generating grounded answer with Ollama..."):
        try:
            answer = generate_answer(question.strip(), chunks, settings)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Answer")
    safe_answer = html.escape(answer).replace("\n", "<br>")
    st.markdown(f'<div class="answer-box">{safe_answer}</div>', unsafe_allow_html=True)

    st.subheader("Sources Used")
    for index, chunk in enumerate(chunks, start=1):
        with st.expander(
            f"{index}. {chunk['citation']} - score {chunk['score']:.3f}",
            expanded=index == 1,
        ):
            st.markdown(
                f'<div class="source-meta">Retrieved from {html.escape(chunk["citation"])}</div>',
                unsafe_allow_html=True,
            )
            st.write(chunk["text"])
