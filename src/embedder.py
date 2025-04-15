from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.create_documents([text])

def store_in_vector_db(docs, persist_dir, model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    FAISS.save_local(vectordb, persist_dir)
    return vectordb
