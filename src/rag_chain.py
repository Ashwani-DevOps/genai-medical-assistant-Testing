import os
import pickle
from langchain.vectorstores import FAISS

def retrieve_context(query, index_path="embeddings/faiss_index.pkl"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ FAISS index not found at: {index_path}")
    
    with open(index_path, "rb") as f:
        db = pickle.load(f)
    
    return db.similarity_search(query, k=3)