import streamlit as st
import pandas as pd
import tempfile
import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# --- Config ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LLM_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Excel Chatbot", page_icon="📊")
st.title("📊 Chat with your Excel")

# Cache the embedding model so it loads once, not on every interaction
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


def build_index(chunks, collection_name):
    db = chromadb.Client()  # in-memory, fresh per session
    try:
        db.delete_collection(collection_name)
    except:
        pass
    collection = db.create_collection(collection_name)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection


def ask(question, collection, n_results=20):
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=n_results)
    context = "\n".join(results["documents"][0])

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


# --- Session state to remember things between user actions ---
if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filename" not in st.session_state:
    st.session_state.filename = None

# --- File uploader ---
uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded and uploaded.name != st.session_state.filename:
    # New file uploaded — process it
    with st.spinner(f"Indexing {uploaded.name}..."):
        # Save to a temp file since pandas needs a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        chunks = excel_to_chunks(tmp_path)
        os.unlink(tmp_path)
        st.session_state.collection = build_index(chunks, f"col_{hash(uploaded.name)}")
        st.session_state.filename = uploaded.name
        st.session_state.messages = []  # reset chat for new file
    st.success(f"Indexed {len(chunks)} rows from {uploaded.name}. Ask away!")

# --- Chat interface ---
if st.session_state.collection is not None:
    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(prompt, st.session_state.collection)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload an Excel file to start chatting.")