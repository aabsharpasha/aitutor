import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from src import config

# Set OpenAI API key
openai_api_key = config.OPEN_API_KEY  # Keep this secure in production

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the vector database
vector_db = Chroma(
    persist_directory=config.CHROMA_DB_DIR,
    embedding_function=embedding_model,
)

# Initialize the language model
llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    chain_type="stuff"
)

# Function to ask a rephrased question
def ask_question(query, length, style, level):
    try:
        # Step 1: Retrieve documents
        retriever = vector_db.as_retriever()
        docs = retriever.get_relevant_documents(query)

        # Step 2: Build a context string from retrieved docs
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: Build the final prompt to LLM
          # Full rephrase instruction to include all dropdown selections
        full_prompt = (
            f"You are an AI tutor. Based on the following context, answer the question,"
            f"in your own words, using a different style than the book,"
            f"{length.lower()} format, styled as {style.lower()}, and suitable for a {level.lower()} learner.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )

        # Step 4: Call LLM directly (not through RetrievalQA)
        response = llm.invoke(full_prompt)

        return {
            "retrieved_docs": context,
            "final_prompt": full_prompt,
            "llm_response": response
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# Streamlit frontend
def run_streamlit_app():
    

    st.title("📘 AI Tutor - Ask Your Questions")

    question = st.text_input("❓ Ask a question about the chapter")
    #length = st.selectbox("Select Answer Type:", ["Short", "Long"])
    #style = st.selectbox("Select Answer Style:", ["Elaborated", "With Examples", "Bullet Points"])
    #level = st.selectbox("Understanding Level:", ["Beginner", "Intermediate", "Advanced"])
    
    
    col1, col2, col3 = st.columns(3)

    with col1:
        length = st.selectbox("Select Answer Type:", ["Short", "Long"])

    with col2:
        style = st.selectbox("Select Answer Style:", ["Elaborated", "With Examples", "Bullet Points"])

    with col3:
        level = st.selectbox("Understanding Level:", ["Beginner", "Intermediate", "Advanced"])


    if question:
        # Add rephrasing instruction to the user query
        #rephrased_prompt = f"Answer this in your own words, using a different style than the book: {question}"
        # rephrase_instruction = (
        #     f"with a {length.lower()} length, "
        #     f"styled as {style.lower()}, "
        #     f"and suitable for a {level.lower()} learner. "
        #     f"Here is the question: {question}"
        # )


        with st.spinner("Thinking..."):
            result = ask_question(question, length, style, level)
            
        if "error" in result:
         st.error(result["error"])
        else:
         if st.checkbox("Show debug info (retrieved docs and prompt)"):
            st.subheader("🔍 Retrieved Content:")
            st.text(result["retrieved_docs"])

            st.subheader("🧠 Final Prompt to LLM:")
            st.text(result["final_prompt"])

         st.subheader("💬 Answer:")
         st.write(result["llm_response"])

         #st.markdown("### 🧠 Answer")
        #st.write(answer)

# Run the app
if __name__ == "__main__":
    run_streamlit_app()
