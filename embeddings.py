from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Free model
    embeddings = model.encode(chunks)
    return embeddings

def save_to_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, "faiss_index.bin")  # Save for reuse

# Example usage  
if __name__ == "__main__":
    from document_loader import extract_text
    text = extract_text("data/policy.pdf")
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    save_to_faiss(embeddings)