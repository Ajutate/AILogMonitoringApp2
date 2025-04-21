# === vector_store.py ===
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from log_processor import process_logs

def build_vector_store():
    documents = process_logs("application_logs.txt")
    if not documents:
        raise ValueError("No valid documents were processed from the log file!")
    
    print(f"Documents processed successfully. {documents}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory="./chroma_logsN")
    vectordb.persist()

# Only run once to build store
if __name__ == "__main__":
    build_vector_store()
    print("Vector store built and persisted successfully.")