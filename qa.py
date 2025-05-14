import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import os
import pickle
from src import config
import boto3
import uuid
import re

from dotenv import load_dotenv

load_dotenv()  # load .env variables

# Now you can access them like this:
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_REGION")

def convertToSpeech(presponse, selected_language):
    # Clean up markdown formatting
    presponse = re.sub(r"\*\*|\*", "", presponse)  # Removes ** and *
    presponse = re.sub(r"_", "", presponse)        # Removes _ if used
    presponse = re.sub(r"\s+", " ", presponse).strip()

    # Language and VoiceId mapping
    language_mapping = {
        "English": {"VoiceId": "Kajal", "LanguageCode": "en-IN"},
        "Hindi": {"VoiceId": "Kajal", "LanguageCode": "hi-IN"},
        "Hinglish": {"VoiceId": "Kajal", "LanguageCode": "hi-IN"},
        # Add more languages as needed
    }

    # Get the corresponding VoiceId and LanguageCode for the selected language
    voice_id = language_mapping.get(selected_language, {"VoiceId": "Joanna", "LanguageCode": "en-US"})["VoiceId"]
    language_code = language_mapping.get(selected_language, {"VoiceId": "Joanna", "LanguageCode": "en-US"})["LanguageCode"]

    # Initialize Polly
    polly = boto3.client("polly",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    # Synthesize speech
    result = polly.synthesize_speech(
        Text=presponse,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine="neural",
        LanguageCode=language_code
    )

    # Save to temp file
    audio_file = f"{uuid.uuid4()}.mp3"
    with open(audio_file, "wb") as f:
        f.write(result["AudioStream"].read())

    # Load audio bytes and store in session
    with open(audio_file, "rb") as f:
        st.session_state.audio_bytes = f.read()

    st.success("✅ Audio generated!")

# Modify the button action to pass the selected language
def handle_speech_button(response, selected_language):
    if response and response.strip():
        convertToSpeech(response, selected_language)
    else:
        st.warning("No response available to convert to speech.")


os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models"
    )
    persist_dir = config.CHROMA_DB_DIR  

    faiss_index_path = os.path.join(persist_dir, "index.faiss")
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index file not found in directory: {persist_dir}")

    db = FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)
    return db

vector_db = load_vector_db()    

llm = ChatOpenAI(
    temperature=0.2,
    api_key=openai_api_key,
    model="gpt-4.1"
)

def ask_question(query, length, style, level, selected_books, lang='English'):
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 10, "filter": {"type": "topics"}})
        docs = retriever.get_relevant_documents(query)
        
        retriever2 = vector_db.as_retriever(search_kwargs={"k": 5, "filter": {"type": "example"}})
        docs2 = retriever2.get_relevant_documents(query)

        if selected_books:
            docs = [doc for doc in docs if doc.metadata.get("source") in selected_books]

        docs_sorted = sorted(docs, key=lambda doc: doc.metadata.get("priority", 100))[:5]

        context_parts = []
        sources = []

        for doc in docs_sorted:
            source = doc.metadata.get("source", "Unknown Source")
            chunk_text = f"[Source: {source}]\n{doc.page_content.strip()}"
            context_parts.append(chunk_text)
            sources.append(source)
            
        for doc2 in docs2:
            source = doc2.metadata.get("source", "Unknown Source")
            chunk_text = f"[Source: {source}]\n{doc2.page_content.strip()}"
            context_parts.append("\nExample Context:\n"+ chunk_text)

        context = "\n\n".join(context_parts)
        unique_sources = list(dict.fromkeys(sources))  

        lang_instruction = ''
        if length == "Very Short":
            word_instruction = "Write an extremely concise explanation in your own words. Your answer must be at least 30 words and not more than 40 words."
        elif length == "Short":
            word_instruction = "Write a concise explanation in your own words. Your answer must be at least 80 words and not more than 90 words."
        else:
            word_instruction = "Write a detailed explanation in your own words. Your response MUST be no fewer than 200 words and MUST NOT exceed 220 words."

        if style == "Concise":
            style_instruction = (
                "Provide a clear and brief explanation without unnecessary elaboration. "
                "Keep it straightforward and to the point."
            )
        elif style == "With Examples":
            style_instruction = (
                "Act like a teacher explaining to a student. "
                "You MUST extract the example strictly from the section titled 'Example Context' in the provided context. "
                "You MUST include at least one clear, relevant example. "
                "Start the example with the word 'Example:' in new line followed by a detailed explanation. "
                "The example should be explained in your own words. "
            )
        elif style == "Bullet Points":
            style_instruction = (
                "Present the explanation using bullet points for clarity. "
                "Ensure the explanation stays within the word limit of the response. "
            )
            
        if lang == "Hinglish":
            lang_instruction = (
                "Please provide output in Hinglish."
            )

        full_prompt = (
            f"IMPORTANT: You MUST follow ALL instructions below exactly.\n"
            f"IMPORTANT: Read the question carefully and answer ALL parts of the question.\n"
            f"You are an AI assistant that must respond strictly based on the provided context.\n"
            f"- Do NOT add, assume, or infer any information that is not explicitly in the context.\n"
            f"- If you're asked to generate example content, only do so when explicitly instructed."
            f"{word_instruction} {style_instruction} {lang_instruction}\n\n"
            f"If the question has more than one part, address EACH part clearly.\n"
            f"If needed, break your answer into labeled sections including heading and subheading to ensure clarity.\n\n"
            f"You are a Virtual Subject Expert. Based on the following context, answer the question "
            f"in your own words, using a different style than the book. "
            f"The explanation should be suitable for a {level.lower()} learner. "
            f"Strictly adhere to the word limit and example instructions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
        )

        if docs_sorted:
            response = llm.invoke(full_prompt)
        else:
            response = "No relevant answer found in the book. Please try a different question."

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

  
def run_streamlit_app():
    st.title("📘 AI Tutor - Ask Your Questions")

    question = st.text_area("❓ Ask a question about the chapter", height=100)

    book_priority_map = {
        'NCERT Theory Sociology +2 - Chp 2.pdf': 1,
        'Mcgraw Hill NCERT Compendium - Chp - 2.pdf': 2,
        'Delhi Government.pdf': 3,
        'Full Marks AK Bhatnagar Theory - Chp 2.pdf': 4,
        'Arihant 10 Sample Theory - Chp 2.pdf': 5,
        'CL Educate Sociology CBSE Study Guide 12 - Chp 2.pdf': 6,
        'Golden Sociology Theory - Chp 2.pdf': 7,
        'Gullybaba The Study of Society 11 - Chp - 2.pdf': 8,
        'Gullybaba Society In India 12 - Chp - 2.pdf': 9,
        'Anand Kumar 12th Sociology - Chp - 2.pdf': 10,
        'CBSE notes National Digital library.pdf': 11,
        'lesy_10204_eContent.pdf': 12
    }

    selected_books = st.multiselect(
        "📚 Select Books to Prioritize (optional):",
        options=list(book_priority_map.keys()),
    )

    col1, col2, col3 , col4 = st.columns(4)
    with col1:
        length = st.selectbox("Select Answer Type:", ["Very Short","Short", "Long"])
    with col2:
        style = st.selectbox("Select Answer Style:", ["Concise", "With Examples", "Bullet Points"])
    with col3:
        level = st.selectbox("Understanding Level:", ["Beginner", "Intermediate", "Advanced"])
    with col4:
        lang = st.selectbox("Choose Language", ["English", "Hinglish"])

    if "response" not in st.session_state:
        st.session_state.response = ""

    if st.button("Get Answer") and question:
        result = ask_question(question, length, style, level, selected_books, lang)

        if "error" in result:
            st.error(f"🚨 Error: {result['error']}")
        else:
            st.subheader("📘 AI Generated Answer:")
            st.write(result["llm_response"])

            st.session_state.response = result["llm_response"]
            
            st.button("🔊 Convert to Speech")
            handle_speech_button(result["llm_response"], lang)

    if st.session_state.get("audio_bytes"):
        st.audio(st.session_state["audio_bytes"], format="audio/mp3")


run_streamlit_app()
