import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from rag_chain import retrieve_context

# Load environment variables
load_dotenv()
LM_API = os.getenv("LMSTUDIO_API")  # e.g. https://weariest-lacklustrely-jessia.ngrok-free.dev
MODEL_NAME = os.getenv("MODEL_NAME")  # e.g. mistral-7b-instruct-v0.2

app = Flask(__name__)

def query_llm(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        res = requests.post(f"{LM_API}/v1/chat/completions", json=payload)
        res.raise_for_status()
        data = res.json()

        # Parse OpenAI-style response
        if "choices" in data and "message" in data["choices"][0]:
            return data["choices"][0]["message"]["content"]
        else:
            print("❌ Unexpected response format:", data)
            return "❌ Error: Unexpected LLM response format"

    except Exception as e:
        print("❌ Error querying LLM:", str(e))
        return f"❌ Error querying LLM: {e}"

@app.route("/medical-suggestion", methods=["GET"])
def medical_suggestion():
    query = request.args.get("prompt", "")
    if not query or len(query) > 300:
        return jsonify({"error": "Invalid or too long query"}), 400

    try:
        docs = retrieve_context(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    except FileNotFoundError:
        return jsonify({"error": "FAISS index not found"}), 500

    prompt = f"""You are a helpful medical assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""
    answer = query_llm(prompt)
    return jsonify({"suggestion": answer})

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "✅ Backend is running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)