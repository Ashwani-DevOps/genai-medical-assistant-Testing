def inject_context(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return f"""You are a medical assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""