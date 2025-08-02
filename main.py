from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Load pre-built FAISS index
index = faiss.read_index("faiss_index.bin")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key="your-api-key")  # Replace with your key

def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]  # Returns chunk IDs

def ask_llm(query, context_chunks):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer using ONLY this context:"},
            {"role": "user", "content": f"Context: {context_chunks}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# Example usage  
if __name__ == "__main__":
    query = "What is the late fee penalty?"
    chunk_ids = search(query)
    with open("data/policy.pdf", "r") as f:
        chunks = f.read().split('\n')  # Simulate loading chunks
    relevant_chunks = [chunks[i] for i in chunk_ids]
    answer = ask_llm(query, "\n".join(relevant_chunks))
    print("Answer:", answer)