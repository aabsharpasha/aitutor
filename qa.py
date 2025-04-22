import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import os
import pickle
from src import config

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def load_vector_db():
    # Correct model name + optional cache folder
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models"  # optional but recommended
    )
    persist_dir = config.CHROMA_DB_DIR  # Reusing the same path for FAISS

    # Check if the index file exists
    faiss_index_path = os.path.join(persist_dir, "index.faiss")
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index file not found in directory: {persist_dir}")

    # Loading FAISS index
    db = FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load once per session
vector_db = load_vector_db()    




# Load API key securely from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Cache the vector DB to avoid reloading on every interaction


# Initialize the language model
#llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
llm = ChatOpenAI(
    temperature=0.7,
    api_key=openai_api_key,
    model="gpt-4.1"
)

# Function to ask a rephrased question
def ask_question(query, length, style, level):
    try:
        retriever = vector_db.as_retriever()
        docs = retriever.get_relevant_documents(query)
        docs = docs[:3]

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

        # full_prompt = (
        #     f"You are an AI tutor. Based on the following context, answer the question "
        #     f"in your own words, using a different style than the book. "
        #     f"{'Write a concise explanation (under 100 words)' if length == 'Short' else 'Write a detailed explanation (at least 300 words)'} in a {style.lower()} style, "
        #     f"suitable for a {level.lower()} learner.\n\n"
        #     f"Context:\n{context}\n\n"
        #     f"Question:\n{query}"
        # )
        
        if length == "Very Short":
            word_instruction = "Write an extremely concise explanation in your own words. Limit your answer strictly to around 30 words."
        elif length == "Short":
            word_instruction = "Write a concise explanation in your own words. Limit your answer strictly to around 80 words."
        else:
            word_instruction = "Write a detailed explanation in your own words. Limit your answer strictly to approximately 200 words."

        if style == "Concise":
             style_instruction = (
                "Provide a clear and brief explanation without unnecessary elaboration. "
                "Keep it straightforward and to the point."
            )
        elif style == "With Examples":
            style_instruction = (
                "You MUST include at least one clear, relevant example. "
                "Start the example with the word 'Example:' in new line followed by a detailed explanation. "
                "If the context lacks examples, create your own. "
                "Not including an example will result in an incomplete answer."
            )
        elif style == "Bullet Points":
            style_instruction = (
                "Present the explanation using bullet points for clarity. "
                "Ensure the explanation stays within the word limit of the response. "
            )

        full_prompt = (
            f"IMPORTANT: You MUST follow ALL instructions below exactly.\n"
            f"IMPORTANT: Read the question carefully and answer ALL parts of the question.\n"
            f"If the question has more than one part, address EACH part clearly.\n"
            f"If needed, break your answer into labeled sections to ensure clarity.\n\n"
            f"You are a Virtual Subject Expert. Based on the following context, answer the question "
            f"in your own words, using a different style than the book. "
            f"{word_instruction} {style_instruction} "
            f"The explanation should be suitable for a {level.lower()} learner. "
            f"Strictly adhere to the word limit and example instructions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
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

    question = st.text_area("❓ Ask a question about the chapter", height=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        length = st.selectbox("Select Answer Type:", ["Very Short","Short", "Long"])
    with col2:
        style = st.selectbox("Select Answer Style:", ["Concise", "With Examples", "Bullet Points"])
    with col3:
        level = st.selectbox("Understanding Level:", ["Beginner", "Intermediate", "Advanced"])

    if st.button("Get Answer") and question:
        result = ask_question(question, length, style, level)

        if "error" in result:
            st.error(f"🚨 Error: {result['error']}")
        else:
            st.subheader("📄 Answer")
            
            response = result["llm_response"].strip()
            
            if response.lower().startswith("answer:"):
                response = response[len("Answer:"):].lstrip()
            
            st.markdown(response)
            
            word_count = len(response.split())
            st.markdown(f"**Word Count:** {word_count}")



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


            #with st.expander("🧠 Final Prompt Sent to LLM"):
            #  st.code(result["final_prompt"], language="markdown")
                
            with st.expander("📚 Sources"):
                for src in result.get("sources", []):
                  st.markdown(f"- `{src}`")


# Run the app
if __name__ == "__main__":
    run_streamlit_app()
