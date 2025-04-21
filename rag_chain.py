# === rag_chain.py ===
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def get_qa_chain(metadata_filter: dict = None):
    prompt = PromptTemplate.from_template("""
You are an intelligent log monitoring assistant analyzing application logs. Your job is to provide accurate answers based STRICTLY on the log entries provided in the context below.

CRITICAL INSTRUCTIONS:
1. ONLY reference log entries that are EXPLICITLY provided in the context
2. NEVER create, invent, or hallucinate log entries that aren't in the context
3. When quoting log entries, use the EXACT format and content as provided
4. If no relevant logs are found in the context, clearly state "No relevant logs found in the provided data" - DO NOT make up examples
5. If asked about a specific time period and no logs from that period are in the context, state "No logs from that time period were found"

You can:
- Count specific types of errors/events (only those present in the context)
- Extract unique messages/patterns (only from the provided logs)
- Summarize events (only from logs in the context)
- Identify root causes (based solely on available logs)

Question: {question}
Log Data:
{context}

Your Answer (remember to ONLY use log entries that appear in the context above):
""")

    print(f"Creating QA chain with metadata filter: {metadata_filter}")
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
    
    # Debug: print collection info
    print(f"Vector DB collection info: {vectordb._collection.count()} documents")
    
    # Fix: Only use filter if it's not empty
    search_kwargs = {"k": 20}
    if metadata_filter and len(metadata_filter) > 0:
        search_kwargs["filter"] = metadata_filter
        print(f"Using filter: {metadata_filter}")
    else:
        print("No filter applied")
    
    # Create retriever with the search kwargs
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    
    # Debug: try a simple retrieval to verify it works
    if metadata_filter:
        try:
            test_docs = retriever.get_relevant_documents("test query")
            print(f"Retrieved {len(test_docs)} documents with filter")
            if test_docs:
                print(f"Sample timestamp: {test_docs[0].metadata.get('timestamp', 'No timestamp')}")
        except Exception as e:
            print(f"Error testing retriever: {e}")
    
    llm = Ollama(model="llama3.2")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  # Return source documents for verification
    )
