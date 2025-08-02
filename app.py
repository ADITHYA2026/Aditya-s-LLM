import streamlit as st
from main import search, ask_llm

st.title("📄 Document Q&A Bot")
query = st.text_input("Ask a question:")

if query:
    chunk_ids = search(query)
    with open("data/policy.pdf", "r") as f:
        chunks = f.read().split('\n')
    relevant_chunks = [chunks[i] for i in chunk_ids]
    answer = ask_llm(query, "\n".join(relevant_chunks))
    st.write("🔍 Answer:", answer)