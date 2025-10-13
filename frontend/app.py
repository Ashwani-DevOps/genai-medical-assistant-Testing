import streamlit as st
import requests

st.set_page_config(page_title="GenAI Medical Assistant", layout="centered")
st.title("🩺 GenAI Medical Assistant")
st.markdown("Ask a medical question and get real-time guidance using LM Studio + FAISS + OpenFDA.")

query = st.text_input("🔍 Enter your medical question:")

if query:
    if len(query) > 300:
        st.error("⚠️ Query too long. Please shorten your question.")
    elif not query.strip().replace(" ", "").isalnum():
        st.error("⚠️ Invalid input. Please use plain text.")
    else:
        with st.spinner("💬 Thinking..."):
            try:
                res = requests.get("http://genai-backend:5000/medical-suggestion", params={"prompt": query})
                data = res.json()
                if "suggestion" in data:
                    st.markdown("### 🧠 Answer")
                    st.write(data["suggestion"])
                else:
                    st.error(f"❌ Error: {data.get('error', 'Unknown issue')}")
            except Exception as e:
                st.error(f"❌ Failed to connect to backend: {e}")