import os
from src import config
from src.pdf_loader import extract_text_from_pdf
from src.embedder import split_text, store_in_vector_db

if __name__ == "__main__":
    pdf_path = os.path.join(config.PDF_DIR, config.PDF_FILE)
    print("📘 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("✂️ Splitting text into chunks...")
    documents = split_text(text)

    print("📦 Storing in vector database...")
    store_in_vector_db(documents, config.CHROMA_DB_DIR, config.EMBEDDING_MODEL)

    print("✅ Done. Embeddings stored.")
