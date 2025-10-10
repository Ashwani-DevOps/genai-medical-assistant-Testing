import os
import pickle
import requests
import streamlit as st
from dotenv import load_dotenv
from rag_chain import retrieve_context

# ✅ Load environment variables
load_dotenv()
LM_API = os.getenv("LMSTUDIO_API")  # ✅ This should be just the URL

# ✅ Query LM Studio using prompt-style payload

def query_llm(prompt):
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 512,
        "model": "mistral-7b-instruct-v0.2"
    }

    try:
        res = requests.post(LM_API, json=payload)
        res.raise_for_status()
        data = res.json()

        if "choices" not in data or not data["choices"]:
            st.error("❌ LLM response missing 'choices'. Check LM Studio response format.")
            st.json(data)
            st.stop()

        return data["choices"][0]["text"]

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request to LM Studio failed: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()


# ✅ Streamlit UI
st.set_page_config(page_title="GenAI Medical Assistant", layout="centered")
st.title("🩺 GenAI Medical Assistant")
st.markdown("Ask a medical question and get real-time guidance using LM Studio + FAISS + OpenFDA.")

query = st.text_input("🔍 Enter your medical question:")

if query:
    # ✅ Validate input
    if len(query) > 300:
        st.error("⚠️ Query too long. Please shorten your question.")
        st.stop()
    if not query.strip().replace(" ", "").isalnum():
        st.error("⚠️ Invalid input. Please use plain text.")
        st.stop()

    # ✅ Retrieve context from FAISS
    try:
        docs = retrieve_context(query)
        context = "\n\n".join([doc.page_content for doc in docs])
    except FileNotFoundError:
        st.error("❌ FAISS index not found. Run ingest_pdf.py first.")
        st.stop()

    # ✅ Build prompt
    prompt = f"""You are a helpful medical assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

    # ✅ Query LLM
    with st.spinner("💬 Thinking..."):
        answer = query_llm(prompt)

    # ✅ Display result
    st.markdown("### 🧠 Answer")
    st.write(answer)