import os
import fitz  # PyMuPDF
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# ✅ Extract text from PDF
def extract_text(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found at: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return texts

# ✅ Create FAISS index
def create_vector_store(texts, index_path):
    docs = [Document(page_content=t) for t in texts]
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedder)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(db, f)
    
    print(f"✅ FAISS index saved to: {index_path}")

# ✅ Run ingestion
if __name__ == "__main__":
    pdf_path = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
    index_path = "embeddings/faiss_index.pkl"

    print("🔍 Extracting text from PDF...")
    texts = extract_text(pdf_path)

    print("📦 Creating FAISS vector store...")
    create_vector_store(texts, index_path)