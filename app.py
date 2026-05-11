import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Config ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
FAST_MODEL = "gpt-4o-mini"   # for routing & SQL writing (cheap)
SMART_MODEL = "gpt-4o"        # for final answer generation (smart)

st.set_page_config(page_title="Excel Chatbot", page_icon="📊")
st.title("📊 Chat with your Excel")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_openai():
    return OpenAI(api_key=OPENAI_API_KEY)


embedder = load_embedder()
openai_client = load_openai()


def safe_name(name):
    """Make a string safe to use as SQL table/column name."""
    s = re.sub(r"\W+", "_", str(name).lower()).strip("_")
    return s or "col"


def load_workbook(filepath):
    """Read every sheet, return cleaned DataFrames and a SQLite DB with the data."""
    sheets = pd.read_excel(filepath, sheet_name=None)
    cleaned = {}
    db_path = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
    conn = sqlite3.connect(db_path)
    for name, df in sheets.items():
        if len(df) < 1 or len(df.columns) < 1:
            continue
        df = df.copy()
        df.columns = [safe_name(c) for c in df.columns]
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if len(df) < 1:
            continue
        cleaned[name] = df
        df.to_sql(safe_name(name), conn, if_exists="replace", index=False)
    conn.close()
    return cleaned, db_path


def excel_to_chunks(sheets):
    chunks = []
    for sheet_name, df in sheets.items():
        headers = df.columns.tolist()
        for idx, row in df.iterrows():
            parts = [f"{h}: {row[h]}" for h in headers if pd.notna(row[h])]
            text = f"Sheet '{sheet_name}', row {idx+1}. " + ". ".join(parts)
            chunks.append(text)
    return chunks


def build_index(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1, norms)


def search(question, chunks, embeddings, n_results=20):
    q_emb = embedder.encode([question])[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)
    scores = embeddings @ q_emb
    top_idx = np.argsort(scores)[::-1][:n_results]
    return [chunks[i] for i in top_idx]


def build_schema_summary(sheets):
    parts = []
    for sheet_name, df in sheets.items():
        table = safe_name(sheet_name)
        col_descs = []
        for col in df.columns:
            sample = df[col].dropna().head(2).tolist()
            col_descs.append(f"  - {col} (sample: {sample})")
        parts.append(f"Table '{table}' ({len(df)} rows):\n" + "\n".join(col_descs))
    return "\n\n".join(parts)


def route(question, schema_summary):
    """LLM decides: NUMERIC (use SQL) or DESCRIPTIVE (use vector search)."""
    resp = openai_client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                f"Schema:\n{schema_summary}\n\n"
                f"Question: {question}\n\n"
                "Reply with exactly one word: NUMERIC if the question needs exact "
                "filtering, counting, summing, averaging, ranking, or aggregating. "
                "DESCRIPTIVE if it asks for summaries, patterns, or open-ended insights."
            )
        }]
    )
    return resp.choices[0].message.content.strip().upper()


def sql_answer(question, schema_summary, db_path):
    # Step 1: cheaper model writes the SQL
    sql_resp = openai_client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Schema:\n{schema_summary}\n\n"
                f"Write a single SQLite query to answer: {question}\n"
                "Reply with ONLY the SQL, no markdown fences, no explanation."
            )
        }]
    )
    sql = sql_resp.choices[0].message.content.strip()
    sql = re.sub(r"^```\w*\n?", "", sql).rstrip("`").strip()

    if re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE)\b", sql, re.I):
        return f"⚠️ Refused to run a query that modifies data. Tried: `{sql}`"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        result = pd.read_sql(sql, conn)
    except Exception as e:
        conn.close()
        return f"⚠️ SQL error: {e}\n\nQuery tried:\n```\n{sql}\n```"
    conn.close()

    # Step 2: smarter model phrases the result
    final = openai_client.chat.completions.create(
        model=SMART_MODEL,
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"SQL used: {sql}\n"
                f"Result:\n{result.to_string(index=False)}\n\n"
                "Write a clear, helpful answer in plain English. Use the result above "
                "as the source of truth. If the data hints at something interesting "
                "(an outlier, a pattern, a notable gap), briefly mention it."
            )
        }]
    )
    return final.choices[0].message.content


def vector_answer(question, chunks, embeddings, schema_summary):
    relevant = search(question, chunks, embeddings)
    context = "\n".join(relevant)
    # Smarter model handles the descriptive answer
    response = openai_client.chat.completions.create(
        model=SMART_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Schema reference:\n{schema_summary}\n\n"
                f"Relevant rows:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer using only what's shown above. If the answer isn't here, say so. "
                "Where helpful, point out patterns or relationships you notice in the data."
            )
        }]
    )
    return response.choices[0].message.content


def ask(question, state):
    path = route(question, state["schema_summary"])
    if "NUMERIC" in path:
        answer = sql_answer(question, state["schema_summary"], state["db_path"])
        return answer, "🔢 SQL"
    else:
        answer = vector_answer(question, state["chunks"], state["embeddings"], state["schema_summary"])
        return answer, "🔍 Vector search"


# --- Session state ---
for key in ("state", "messages", "filename"):
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.messages is None:
    st.session_state.messages = []

# --- File uploader ---
uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded and uploaded.name != st.session_state.filename:
    with st.spinner(f"Indexing {uploaded.name}..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        sheets, db_path = load_workbook(tmp_path)
        os.unlink(tmp_path)
        chunks = excel_to_chunks(sheets)
        embeddings = build_index(chunks)
        schema_summary = build_schema_summary(sheets)
        st.session_state.state = {
            "sheets": sheets,
            "chunks": chunks,
            "embeddings": embeddings,
            "db_path": db_path,
            "schema_summary": schema_summary,
        }
        st.session_state.filename = uploaded.name
        st.session_state.messages = []
    st.success(f"Indexed {len(chunks)} rows across {len(sheets)} sheet(s). Ask away!")

# --- Chat interface ---
if st.session_state.state is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, path_used = ask(prompt, st.session_state.state)
            st.markdown(answer)
            st.caption(f"_Answered via: {path_used}_")
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload an Excel file to start chatting.")