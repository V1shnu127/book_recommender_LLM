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

def recommend_books(query, top_k=5):
    query = query.strip()

    # 1️⃣ Exact title match
    exact = exact_title_match(df, query)
    if len(exact) > 0:
        return format_results(exact.head(top_k), note="Exact title match")

    # 2️⃣ Partial title match
    partial = partial_title_match(df, query)
    if len(partial) > 0:
        return format_results(partial.head(top_k), note="Partial title match")

    # 3️⃣ Author match
    author_hits = df[df["author"].str.lower().str.contains(query.lower())]
    if len(author_hits) > 0:
        return format_results(author_hits.head(top_k), note="Author match")

    # 4️⃣ Semantic search (FAISS)
    return semantic_search(query, top_k)
"""
        )

    return "\n\n".join(results)

# import pandas as pd
# import numpy as np
# import faiss
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset

# # --------------------------------------------------
# # DATA LOADING (FROM HUGGING FACE - KAGGLE MIRROR)
# # --------------------------------------------------

# @st.cache_resource
# def load_books_dataframe():
#     """
#     Loads the book dataset from Hugging Face (public).
#     This mirrors the Kaggle dataset and avoids GitHub size limits.
#     """
#     dataset = load_dataset(
#         "VsquareSheremetyevo/goodreads_books",
#         split="train"
#     )
#     df = dataset.to_pandas()

#     # Standard cleanup
#     df = df.dropna(subset=["title", "author"])
#     df["description"] = df["description"].fillna("")
#     df["genres"] = df.get("genres", "").fillna("")

#     # Normalize text
#     df["title"] = df["title"].astype(str)
#     df["author"] = df["author"].astype(str)

#     return df.reset_index(drop=True)


# df = load_books_dataframe()

# # --------------------------------------------------
# # EMBEDDING + FAISS INDEX
# # --------------------------------------------------

# def build_embedding_text(row):
#     """
#     Rich text representation to improve
#     title + author accuracy
#     """
#     return (
#         f"Title: {row['title']}. "
#         f"Author: {row['author']}. "
#         f"Genres: {row.get('genres', '')}. "
#         f"Description: {row['description']}"
#     )


# @st.cache_resource
# def load_or_create_faiss():
#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     texts = df.apply(build_embedding_text, axis=1).tolist()

#     embeddings = model.encode(
#         texts,
#         show_progress_bar=True,
#         normalize_embeddings=True
#     ).astype("float32")

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embeddings)

#     return model, index, embeddings


# model, index, embeddings = load_or_create_faiss()

# # --------------------------------------------------
# # SEARCH HELPERS
# # --------------------------------------------------

# def exact_title_match(query):
#     q = query.lower().strip()
#     return df[df["title"].str.lower() == q]


# def partial_title_match(query):
#     q = query.lower().strip()
#     return df[df["title"].str.lower().str.contains(q)]


# def author_match(query):
#     q = query.lower().strip()
#     return df[df["author"].str.lower().str.contains(q)]


# def semantic_search(query, top_k=5):
#     query_embedding = model.encode(
#         [query],
#         normalize_embeddings=True
#     ).astype("float32")

#     scores, indices = index.search(query_embedding, top_k)

#     results = df.iloc[indices[0]].copy()
#     results["score"] = scores[0]

#     return results


# # --------------------------------------------------
# # RESULT FORMATTER
# # --------------------------------------------------

# def format_results(results, note=None):
#     output = ""
#     if note:
#         output += f"**{note}**\n\n"

#     for _, row in results.iterrows():
#         output += f"### 📘 {row['title']}\n"
#         output += f"**Author:** {row['author']}\n\n"

#         if row["description"]:
#             output += f"{row['description'][:500]}...\n\n"

#         output += "---\n"

#     return output


# # --------------------------------------------------
# # MAIN RECOMMENDER FUNCTION (USED BY app.py)
# # --------------------------------------------------

# def recommend_books(query, top_k=5):
#     """
#     Hybrid retrieval pipeline:
#     1. Exact title
#     2. Partial title
#     3. Author match
#     4. Semantic FAISS search
#     """

#     query = query.strip()

#     # 1️⃣ Exact title match (fixes "Hooked")
#     exact = exact_title_match(query)
#     if not exact.empty:
#         return format_results(exact.head(top_k), "Exact title match")

#     # 2️⃣ Partial title match
#     partial = partial_title_match(query)
#     if not partial.empty:
#         return format_results(partial.head(top_k), "Partial title match")

#     # 3️⃣ Author match
#     author_hits = author_match(query)
#     if not author_hits.empty:
#         return format_results(author_hits.head(top_k), "Author match")

#     # 4️⃣ Semantic search
#     semantic_results = semantic_search(query, top_k)
#     return format_results(semantic_results, "Semantic recommendations")
