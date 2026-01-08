# import os
# import faiss
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# # =====================================================
# # Paths (absolute — Streamlit safe)
# # =====================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.join(BASE_DIR, "books.csv")
# INDEX_PATH = os.path.join(BASE_DIR, "books.index")

# # =====================================================
# # Load and clean dataset
# # =====================================================
# df = pd.read_csv(
#     CSV_PATH,
#     engine="python",
#     on_bad_lines="skip"
# )

# # Normalize column names
# df.columns = [c.strip().lower() for c in df.columns]

# # Keep only required columns (safety)
# REQUIRED_COLS = [
#     "title", "authors", "publisher",
#     "average_rating", "language_code"
# ]
# df = df[[c for c in REQUIRED_COLS if c in df.columns]]

# df.dropna(inplace=True)

# # =====================================================
# # Create semantic text (VERY IMPORTANT)
# # =====================================================
# df["semantic_text"] = (
#     "Title: " + df["title"].astype(str) +
#     ". Author(s): " + df["authors"].astype(str) +
#     ". Publisher: " + df["publisher"].astype(str) +
#     ". Language: " + df["language_code"].astype(str)
# )

# # =====================================================
# # Embedding model
# # =====================================================
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # =====================================================
# # Load or create FAISS index
# # =====================================================
# def load_or_create_index():
#     if os.path.exists(INDEX_PATH):
#         print("✅ Loading existing FAISS index")
#         return faiss.read_index(INDEX_PATH)

#     print("⚙️ FAISS index not found — creating new one")

#     embeddings = embedder.encode(
#         df["semantic_text"].tolist(),
#         show_progress_bar=True
#     ).astype("float32")

#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     faiss.write_index(index, INDEX_PATH)
#     print("✅ FAISS index created and saved")

#     return index

# index = load_or_create_index()

# # =====================================================
# # Public API — Streamlit imports THIS
# # =====================================================
# def recommend_books(query: str, top_k: int = 5) -> str:
#     """
#     Returns top-k semantically similar books for a user query
#     """
#     query_vec = embedder.encode([query]).astype("float32")
#     _, indices = index.search(query_vec, top_k)

#     results = []
#     for i in indices[0]:
#         results.append(
#             f"""📘 **{df.iloc[i]['title']}**
# 👤 Author(s): {df.iloc[i]['authors']}
# ⭐ Rating: {df.iloc[i]['average_rating']}
# 🏢 Publisher: {df.iloc[i]['publisher']}

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

