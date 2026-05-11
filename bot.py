import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# --- Config ---
GROQ_API_KEY = "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"
EXCEL_FILE = "test.xlsx"
LLM_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


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
    db = chromadb.PersistentClient(path="./chroma_db")
    try:
        db.delete_collection("excel_data")
    except:
        pass
    collection = db.create_collection("excel_data")

    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection


def ask(question, collection):
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=5)
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


# --- Main ---
print("Loading Excel and building index...")
chunks = excel_to_chunks(EXCEL_FILE)
collection = build_index(chunks)
print(f"Indexed {len(chunks)} rows. Ask away! (type 'quit' to exit)\n")

while True:
    question = input("You: ")
    if question.lower() in ("quit", "exit"):
        break
    answer = ask(question, collection)
    print(f"\nBot: {answer}\n")