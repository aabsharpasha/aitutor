import os
from src import config
from src.pdf_loader import extract_text_from_pdf
from src.embedder import split_text, store_in_vector_db

pdf_dir = config.PDF_DIR
chroma_db_dir = config.CHROMA_DB_DIR
embedding_model = config.EMBEDDING_MODEL

if __name__ == "__main__":

    
    all_documents = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"📘 Extracting text from: {filename}")
            text = extract_text_from_pdf(pdf_path)

            print("✂️ Splitting text into chunks...")
            documents = split_text(text)

            # Add metadata to each chunk
            for doc in documents:
                doc.metadata = {"source": filename}

            all_documents.extend(documents)  # collect all

    print("📦 Storing ALL documents in vector database...")
    store_in_vector_db(all_documents, chroma_db_dir, embedding_model)
    print("✅ All PDFs processed and stored.")

