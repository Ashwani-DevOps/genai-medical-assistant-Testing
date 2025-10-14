import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def retrieve_context(query, index_path="/app/embeddings/faiss_index"):
    # Check if both required files exist
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError(f"❌ FAISS index files not found in: {index_path}")

    # Load embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    return db.similarity_search(query, k=3)