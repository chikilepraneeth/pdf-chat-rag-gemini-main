# src/app.py
import os
import re
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from google.api_core.exceptions import NotFound, PermissionDenied, InvalidArgument


# -----------------------------
# Page + Setup
# -----------------------------
st.set_page_config(page_title="PDF RAG Chat (Gemini)", layout="wide")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing API key. Add GEMINI_API_KEY (or GOOGLE_API_KEY) to your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Helpers: PDF + Chunking
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_paragraphs(text: str, min_chars: int = 250) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    chunks = []
    buf = ""
    for p in raw_paras:
        if len(buf) < min_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            chunks.append(buf)
            buf = p

    if buf:
        chunks.append(buf)

    return chunks


# -----------------------------
# Gemini model selection + fallback
# -----------------------------
@st.cache_data(show_spinner=False)
def list_working_gemini_models() -> list[str]:
    models = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            models.append(m.name)
    return models


def ask_gemini_with_fallback(prompt: str, model_candidates: list[str]) -> tuple[str, str]:
    last_err = None
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if text and text.strip():
                return text.strip(), model_name
            last_err = RuntimeError(f"Empty response from {model_name}")
        except (NotFound, PermissionDenied, InvalidArgument) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("No available Gemini model worked.")


# -----------------------------
# RAG: Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    m.max_seq_length = 256
    return m


def build_faiss_index(chunks: list[str], embedder: SentenceTransformer):
    emb = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index


def retrieve(chunks: list[str], index: faiss.IndexFlatL2, embedder: SentenceTransformer, query: str, k: int = 4):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, I = index.search(q, k)
    return [chunks[i] for i in I[0] if 0 <= i < len(chunks)]


# -----------------------------
# UI Header
# -----------------------------
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom: 0.2rem;">📄 Chat with Your PDF</h1>
    <p style="text-align:center; margin-top: 0; opacity: 0.8;">Gemini + RAG (SentenceTransformers + FAISS)</p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-k chunks", min_value=2, max_value=8, value=4, step=1)
    show_context = st.checkbox("Show retrieved chunks", value=True)

    st.divider()
    st.caption("Gemini models available to your key:")
    model_candidates = list_working_gemini_models()
    if model_candidates:
        st.write("\n".join(model_candidates[:8]))
        if len(model_candidates) > 8:
            st.caption(f"+ {len(model_candidates) - 8} more...")
    else:
        st.error("No Gemini models (generateContent) available.")
        st.stop()

# -----------------------------
# Upload PDF
# -----------------------------
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if not uploaded:
    st.stop()

# Cache per uploaded file (Streamlit keeps this per session)
if "pdf_ready" not in st.session_state or st.session_state.get("pdf_name") != uploaded.name:
    with st.spinner("Reading PDF and building index..."):
        raw_text = extract_pdf_text(uploaded)
        if not raw_text:
            st.warning("No extractable text found in this PDF (might be scanned).")
            st.stop()

        chunks = chunk_paragraphs(raw_text, min_chars=250)
        embedder = load_embedder()
        index = build_faiss_index(chunks, embedder)

        st.session_state.pdf_ready = True
        st.session_state.pdf_name = uploaded.name
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.embedder = embedder

        # Reset chat for new PDF
        st.session_state.messages = []

st.caption(f"PDF loaded: {st.session_state.pdf_name} • Chunks: {len(st.session_state.chunks)}")


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta") and show_context:
            meta = msg["meta"]
            with st.expander("Retrieved chunks (click to open)"):
                st.caption(f"Model used: {meta.get('model_used', 'unknown')} • Top-k: {meta.get('k', '')}")
                for i, c in enumerate(meta.get("retrieved", []), start=1):
                    with st.expander(f"Chunk {i}"):
                        st.write(c)


# -----------------------------
# Chat Input (Enter to Send)
# -----------------------------
user_q = st.chat_input("Ask something about the PDF…")

if user_q:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # Retrieve context
    chunks = st.session_state.chunks
    index = st.session_state.index
    embedder = st.session_state.embedder

    retrieved = retrieve(chunks, index, embedder, user_q, k=k)
    context = "\n\n---\n\n".join(retrieved)

    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I can't find that in the document."

Context:
{context}

Question: {user_q}
Answer:"""

    # Thinking animation while model runs
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, used_model = ask_gemini_with_fallback(prompt, model_candidates)
            except Exception as e:
                answer = f"Sorry — I couldn't generate an answer. Error: {e}"
                used_model = "unknown"

        st.write(answer)

        if show_context:
            with st.expander("Retrieved chunks (click to open)"):
                st.caption(f"Model used: {used_model} • Top-k: {k}")
                for i, c in enumerate(retrieved, start=1):
                    with st.expander(f"Chunk {i}"):
                        st.write(c)

    # Save assistant message + metadata
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": {
                "model_used": used_model,
                "k": k,
                "retrieved": retrieved,
            },
        }
    )
