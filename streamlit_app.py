import streamlit as st
from query import search_chunks, generate_answer

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG Assistant")

# Interface Streamlit
question = st.text_input("❓ Pose ta question:", placeholder="Qu'est-ce que FastAPI?")

if question:
    with st.spinner("🔍 Recherche en cours..."):
        chunks = search_chunks(question)
    
    if chunks:
        # Génération
        with st.spinner("📝 Génération de la réponse..."):
            answer = generate_answer(question, chunks)
        
        # Affichage
        st.markdown("---")
        st.markdown("### 💬 Réponse")
        st.markdown(answer)
        
        st.markdown("---")
        st.markdown("### 📚 Sources utilisées")
        for i, (content, source, similarity) in enumerate(chunks, 1):
            with st.expander(f"{i}. {source} (similarity: {similarity:.2f})"):
                st.write(content[:500] + "..." if len(content) > 500 else content)
    else:
        st.warning("❌ Aucun chunk trouvé")