import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import traceback

# Load environment variables
load_dotenv()
LM_API = os.getenv("LMSTUDIO_API")
MODEL_NAME = os.getenv("MODEL_NAME")

app = Flask(__name__)

# Load FAISS index at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "embeddings", "faiss_index")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "faiss_index.pkl")
print("Looking for FAISS index at:", FAISS_INDEX_FILE)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = None
try:
    if os.path.exists(FAISS_INDEX_FILE):
        print(f"📄 Found FAISS index file: {FAISS_INDEX_FILE}")
        vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings=embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded successfully")
    else:
        print(f"❌ FAISS index file not found: {FAISS_INDEX_FILE}")
except Exception as e:
    print("❌ Exception while loading FAISS index:", str(e))
    traceback.print_exc()
    vector_store = None

def query_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        res = requests.post(f"{LM_API}/v1/chat/completions", json=payload)
        res.raise_for_status()
        data = res.json()

        message = data.get("choices", [{}])[0].get("message", {})
        return message.get("content", "❌ Error: No content in LLM response")

    except Exception as e:
        print("❌ Error querying LLM:", str(e))
        return f"❌ Error querying LLM: {e}"

@app.route("/medical-suggestion", methods=["GET"])
def medical_suggestion():
    global vector_store
    if vector_store is None:
        print("❌ FAISS index is not loaded in memory")
        return jsonify({"error": "FAISS index not found"}), 500

    query = request.args.get("prompt", "")
    if not query or len(query) > 300:
        print("❌ Invalid query:", query)
        return jsonify({"error": "Invalid or too long query"}), 400

    try:
        print("🔍 Received query:", query)
        docs = vector_store.similarity_search(query)
        print("✅ Retrieved docs:", [doc.page_content[:50] for doc in docs])
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print("❌ Error during similarity_search:", str(e))
        traceback.print_exc()
        return jsonify({"error": "FAISS index not found"}), 500

    prompt = f"""You are a helpful medical assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""
    answer = query_llm(prompt)
    return jsonify({"suggestion": answer})

@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "ok",
        "faiss_loaded": vector_store is not None
    }
    return jsonify(status), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)