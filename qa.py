import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from src import config

# Load API key securely from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Cache the vector DB to avoid reloading on every interaction
@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embedding_model,
    )

# Load once per session
vector_db = load_vector_db()

# Initialize the language model
llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)

# Function to ask a rephrased question
def ask_question(query, length, style, level):
    try:
        retriever = vector_db.as_retriever()
        docs = retriever.get_relevant_documents(query)

        # Build context and collect sources
        context_parts = []
        sources = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            chunk_text = f"[Source: {source}]\n{doc.page_content.strip()}"
            context_parts.append(chunk_text)
            sources.append(source)


        context = "\n\n".join(context_parts)
        unique_sources = list(set(sources))  # To avoid duplicates

        full_prompt = (
            f"You are an AI tutor. Based on the following context, answer the question "
            f"in your own words, using a different style than the book. "
            f"{'Write a concise explanation (under 100 words)' if length == 'Short' else 'Write a detailed explanation (at least 300 words)'} in a {style.lower()} style, "
            f"suitable for a {level.lower()} learner.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )


        response = llm.invoke(full_prompt)

        if hasattr(response, "content"):
            response = response.content

        return {
            "retrieved_docs": context,
            "final_prompt": full_prompt,
            "llm_response": response,
            "sources": unique_sources
        }

    except Exception as e:
        return {"error": str(e)}

  
# Streamlit frontend
def run_streamlit_app():
    st.title("📘 AI Tutor - Ask Your Questions")

    question = st.text_input("❓ Ask a question about the chapter")

    col1, col2, col3 = st.columns(3)
    with col1:
        length = st.selectbox("Select Answer Type:", ["Short", "Long"])
    with col2:
        style = st.selectbox("Select Answer Style:", ["Elaborated", "With Examples", "Bullet Points"])
    with col3:
        level = st.selectbox("Understanding Level:", ["Beginner", "Intermediate", "Advanced"])

    if st.button("Get Answer") and question:
        result = ask_question(question, length, style, level)

        if "error" in result:
            st.error(f"🚨 Error: {result['error']}")
        else:
            st.subheader("📄 Answer")
            st.write(result["llm_response"])

            with st.expander("🔍 Retrieved Context with Sources"):
              chunks = result["retrieved_docs"].split("[Source: ")
              for chunk in chunks:
                if not chunk.strip():
                    continue
                lines = chunk.strip().split("\n", 1)
                source = lines[0].strip("]") if len(lines) > 1 else "Unknown Source"
                content = lines[1].strip() if len(lines) > 1 else ""
                st.markdown(f"**📄 Source:** `{source}`")
                st.write(content[:1000] + "..." if len(content) > 1000 else content)
                st.markdown("---")


            with st.expander("🧠 Final Prompt Sent to LLM"):
                st.code(result["final_prompt"], language="markdown")
                
            with st.expander("📚 Sources"):
                for src in result.get("sources", []):
                  st.markdown(f"- `{src}`")


# Run the app
if __name__ == "__main__":
    run_streamlit_app()
