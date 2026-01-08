# import faiss
# import numpy as np
# import pandas as pd
# import os
# from sentence_transformers import SentenceTransformer
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# # Load resources
# df = pd.read_csv(
#     "books.csv",
#     engine="python",
#     sep=",",
#     quotechar='"',
#     escapechar="\\",
#     on_bad_lines="skip"
# )

# index = faiss.read_index("books.index")
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# client = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))

# def recommend_books(query, top_k=3):
#     query_vec = embedder.encode([query]).astype("float32")
#     _, indices = index.search(query_vec, top_k)

#     retrieved = "\n".join(
#         f"- {df.iloc[i]['title']}: {df.iloc[i]['description']}"
#         for i in indices[0]
#     )

#     prompt = f"""
# You are a book recommendation assistant.

# User query:
# {query}

# Candidate books:
# {retrieved}

# Select and explain the best recommendations briefly.
# """

#     response = client.chat.completions.create(
#         model="llama3-70b-8192",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3
#     )

#     return response.choices[0].message.content

# import os
# import faiss
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# # ---------- Paths ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.join(BASE_DIR, "books.csv")
# INDEX_PATH = os.path.join(BASE_DIR, "books.index")

# # ---------- Load data ----------
# df = pd.read_csv(
#     CSV_PATH,
#     engine="python",
#     on_bad_lines="skip"
# )
# df.dropna(inplace=True)

# # ---------- Embedding model ----------
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # ---------- Load or create FAISS index ----------
# def load_or_create_index():
#     if os.path.exists(INDEX_PATH):
#         return faiss.read_index(INDEX_PATH)

#     embeddings = embedder.encode(
#         df["description"].tolist(),
#         show_progress_bar=True
#     ).astype("float32")

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     faiss.write_index(index, INDEX_PATH)
#     return index

# index = load_or_create_index()

# # ---------- PUBLIC FUNCTION (THIS IS WHAT STREAMLIT IMPORTS) ----------
# def recommend_books(query: str, top_k: int = 3) -> str:
#     query_vec = embedder.encode([query]).astype("float32")
#     _, indices = index.search(query_vec, top_k)

#     results = []
#     for i in indices[0]:
#         results.append(
#             f"📘 {df.iloc[i]['title']}: {df.iloc[i]['description']}"
#         )

#     return "\n\n".join(results)


import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================================================
# Paths (absolute — Streamlit safe)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "books.csv")
INDEX_PATH = os.path.join(BASE_DIR, "books.index")

# =====================================================
# Load and clean dataset
# =====================================================
df = pd.read_csv(
    CSV_PATH,
    engine="python",
    on_bad_lines="skip"
)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Keep only required columns (safety)
REQUIRED_COLS = [
    "title", "authors", "publisher",
    "average_rating", "language_code"
]
df = df[[c for c in REQUIRED_COLS if c in df.columns]]

df.dropna(inplace=True)

# =====================================================
# Create semantic text (VERY IMPORTANT)
# =====================================================
df["semantic_text"] = (
    "Title: " + df["title"].astype(str) +
    ". Author(s): " + df["authors"].astype(str) +
    ". Publisher: " + df["publisher"].astype(str) +
    ". Language: " + df["language_code"].astype(str)
)

# =====================================================
# Embedding model
# =====================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# Load or create FAISS index
# =====================================================
def load_or_create_index():
    if os.path.exists(INDEX_PATH):
        print("✅ Loading existing FAISS index")
        return faiss.read_index(INDEX_PATH)

    print("⚙️ FAISS index not found — creating new one")

    embeddings = embedder.encode(
        df["semantic_text"].tolist(),
        show_progress_bar=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print("✅ FAISS index created and saved")

    return index

index = load_or_create_index()

# =====================================================
# Public API — Streamlit imports THIS
# =====================================================
def recommend_books(query: str, top_k: int = 5) -> str:
    """
    Returns top-k semantically similar books for a user query
    """
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        results.append(
            f"""📘 **{df.iloc[i]['title']}**
👤 Author(s): {df.iloc[i]['authors']}
⭐ Rating: {df.iloc[i]['average_rating']}
🏢 Publisher: {df.iloc[i]['publisher']}
"""
        )

    return "\n\n".join(results)
