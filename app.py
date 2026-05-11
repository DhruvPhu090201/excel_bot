import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Config ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LLM_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Excel Chatbot", page_icon="📊")
st.title("📊 Chat with your Excel")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)


embedder = load_embedder()
groq_client = load_groq()


def excel_to_chunks(filepath):
    chunks = []
    sheets = pd.read_excel(filepath, sheet_name=None)
    for sheet_name, df in sheets.items():
        headers = df.columns.tolist()
        for idx, row in df.iterrows():
            parts = [f"{h}: {row[h]}" for h in headers if pd.notna(row[h])]
            text = f"Sheet '{sheet_name}', row {idx+1}. " + ". ".join(parts)
            chunks.append(text)
    return chunks


def build_index(chunks):
    """Embed chunks and return a numpy array of embeddings."""
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1, norms)


def search(question, chunks, embeddings, n_results=20):
    """Find the most relevant chunks for a question using cosine similarity."""
    q_emb = embedder.encode([question])[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)
    # Cosine similarity = dot product of normalized vectors
    scores = embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:n_results]
    return [chunks[i] for i in top_idx]


def ask(question, chunks, embeddings):
    relevant = search(question, chunks, embeddings)
    context = "\n".join(relevant)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Answer the user's question using ONLY the spreadsheet data below. "
                "If the answer isn't in the data, say so.\n\n"
                f"Data:\n{context}\n\n"
                f"Question: {question}"
            )
        }]
    )
    return response.choices[0].message.content


# --- Session state ---
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = None

# --- File uploader ---
uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded and uploaded.name != st.session_state.filename:
    with st.spinner(f"Indexing {uploaded.name}..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        chunks = excel_to_chunks(tmp_path)
        os.unlink(tmp_path)
        embeddings = build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.filename = uploaded.name
        st.session_state.messages = []
    st.success(f"Indexed {len(chunks)} rows from {uploaded.name}. Ask away!")

# --- Chat interface ---
if st.session_state.chunks is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(prompt, st.session_state.chunks, st.session_state.embeddings)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload an Excel file to start chatting.")