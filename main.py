import os
from src import config
from src.pdf_loader import extract_text_from_pdf
from src.embedder import split_text, store_in_vector_db

pdf_dir = config.PDF_DIR
dir_example = config.PDF_DIR+"/video"
chroma_db_dir = config.CHROMA_DB_DIR
embedding_model = config.EMBEDDING_MODEL

book_priority_map = {
    'NCERT Theory Sociology +2 - Chp 2.pdf': 1,
    'video_transcript_ncert.txt': 0,
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

if __name__ == "__main__":

    all_documents = []
    #files = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            #files.append(filename)
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"📘 Extracting text from: {filename}")
            text = extract_text_from_pdf(pdf_path)

            print("✂️ Splitting text into chunks...")
            documents = split_text(text)

            # Add metadata to each chunk
            for doc in documents:
                doc.metadata = {
                    "type": "topics",
                    "source": filename,
                    "priority": book_priority_map.get(filename, 100)  # default low priority if not mapped
                }
            
            #print(documents); exit;
            all_documents.extend(documents)  # collect all
            
    for filename in os.listdir(dir_example):
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            #files.append(filename)
            pdf_path = os.path.join(dir_example, filename)
            print(f"📘 Extracting text from: {filename}")
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(pdf_path)
            elif filename.endswith(".txt"):  # Corrected 'else if' to 'elif'
                with open(pdf_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            

            print("✂️ Splitting text into chunks...")
            documents = split_text(text)

            # Add metadata to each chunk
            for doc in documents:
                doc.metadata = {
                    "type": "example",
                    "source": filename+" - from Khan Migo Video",
                    "priority": book_priority_map.get(filename, 100)  # default low priority if not mapped
                }
            
            #print(documents); exit;
            all_documents.extend(documents)  # collect all


    #print(files)
    print("📦 Storing ALL documents in vector database...")
    store_in_vector_db(all_documents, chroma_db_dir, embedding_model)
    print("✅ All PDFs processed and stored.")
