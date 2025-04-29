# === vector_store.py ===
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from log_processor import process_logs
import threading

class ChromaVectorStore:
    """
    Singleton class to manage a single instance of the Chroma vector database
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ChromaVectorStore, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self._embedding = OllamaEmbeddings(model="mxbai-embed-large")
            self._persist_directory = "./chroma_logsN"
            self._vectordb = Chroma(persist_directory=self._persist_directory, embedding_function=self._embedding)
            ChromaVectorStore._initialized = True
            print("ChromaVectorStore initialized with a single Chroma instance")

    def get_db(self):
        """Returns the Chroma vector database instance"""
        return self._vectordb
    
    def get_embedding_function(self):
        """Returns the embedding function used by this instance"""
        return self._embedding

    def rebuild_from_documents(self, documents):
        """
        Rebuilds the vector store from the provided documents
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)
        
        # Use the existing embedding function for consistency
        self._vectordb = Chroma.from_documents(
            documents=split_docs, 
            embedding=self._embedding, 
            persist_directory=self._persist_directory
        )
        self._vectordb.persist()
        print("Vector store rebuilt and persisted successfully")

    def add_documents(self, documents):
        """
        Add new documents to the existing vector store
        """
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)
            self._vectordb.add_documents(documents=split_docs)
            self._vectordb.persist()
            print(f"Added {len(documents)} documents to vector store")
        else:
            print("No documents to add")

# Function to get the singleton instance
def get_vector_store():
    """
    Returns the singleton instance of ChromaVectorStore
    """
    return ChromaVectorStore()

def build_vector_store():
    """
    Builds the vector store from scratch (used for initial setup)
    """
    documents = process_logs("application_logs.txt")
    if not documents:
        raise ValueError("No valid documents were processed from the log file!")
    
    print(f"Documents processed successfully. {documents}")
    
    # Get the singleton instance and rebuild from documents
    vector_store = get_vector_store()
    vector_store.rebuild_from_documents(documents)

# Only run once to build store
if __name__ == "__main__":
    build_vector_store()
    print("Vector store built and persisted successfully.")