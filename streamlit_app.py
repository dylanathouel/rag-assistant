import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG Assistant")

with st.sidebar:
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health["status"] == "ok":
            st.success(f"✅ {health['documents']} docs / {health['chunks']} chunks")
        else:
            st.error(f"API: {health.get('detail')}")
    except Exception as e:
        st.error(f"API unreachable: {e}")
    top_k = st.slider("Top-K chunks", 1, 20, 5)

question = st.text_input("❓ Question :", placeholder="What is FastAPI?")

if question:
    with st.spinner("🔍 Recherche + génération..."):
        try:
            data = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=60,
            ).json()
        except Exception as e:
            st.error(f"Erreur API: {e}")
            st.stop()

    st.markdown("### 💬 Réponse")
    st.markdown(data["answer"])

    st.markdown("### 📚 Sources")
    for i, src in enumerate(data["sources"], 1):
        label = src["title"] or src["source"]
        with st.expander(f"{i}. {label} (sim: {src['similarity']:.3f})"):
            st.write(src["content_preview"])
            st.caption(f"source: {src['source']}")