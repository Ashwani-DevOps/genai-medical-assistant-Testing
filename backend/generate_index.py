import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ✅ Sample medical documents — replace or extend with real data later
docs = [
    Document(page_content="Ibuprofen is used to reduce fever and relieve pain or inflammation."),
    Document(page_content="Amlodipine is prescribed for high blood pressure and chest pain."),
    Document(page_content="Paracetamol is commonly used for mild pain and fever relief."),
    Document(page_content="Metformin is used to control high blood sugar in people with type 2 diabetes."),
    Document(page_content="Atorvastatin is used to improve cholesterol levels and reduce the risk of heart disease."),
]

# ✅ Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create FAISS vector store
db = FAISS.from_documents(docs, embedding_model)

# ✅ Save index to folder
output_dir = "embeddings/faiss_index"
os.makedirs(output_dir, exist_ok=True)
db.save_local(output_dir)

print(f"✅ FAISS index saved to: {output_dir}")