import streamlit as st
from recommender import recommend_books

st.set_page_config(
    page_title="Semantic Book Recommender",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Semantic Book Recommender")
st.write(
    "This app uses **semantic search + Groq LLM** to recommend books "
    "based on meaning, not keywords."
)

query = st.text_input(
    "Describe what kind of book you want to read:",
    placeholder="e.g. books about artificial intelligence and neural networks"
)

if st.button("Get Recommendations"):
    if query.strip() == "":
        st.warning("Please enter a description.")
    else:
        with st.spinner("Finding best matches..."):
            result = recommend_books(query)
        st.success("Here are your recommendations:")
        st.write(result)
