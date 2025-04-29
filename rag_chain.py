# === rag_chain.py ===
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def get_qa_chain(metadata_filter: dict = None):
    prompt = PromptTemplate.from_template("""
You are an intelligent log monitoring assistant analyzing application logs. Your job is to provide accurate answers based STRICTLY on the log entries provided in the context below.

CRITICAL INSTRUCTIONS:
1. ONLY reference log entries that are EXPLICITLY provided in the context
2. NEVER create, invent, or hallucinate log entries that aren't in the context
3. When quoting log entries, use the EXACT format and content as provided
4. If no relevant logs are found in the context, clearly state "No relevant logs found in the provided data" - DO NOT make up examples
5. If asked about a specific time period and no logs from that period are in the context, state "No logs from that time period were found"

SPECIAL HANDLING FOR COUNTING QUERIES:
1. When asked to count log entries (e.g., "How many ERROR/ERRORS/EXCEPTION/EXCEPTIONS logs are there?"):
   - Count ALL matching log entries in the context VERY CAREFULLY
   - Start your response with the TOTAL COUNT in a clear sentence
   - Then show 3-5 examples of the logs as evidence
   - Format examples with the exact timestamp and message
2. Double-check your counting to ensure accuracy
3. Be precise with your numbers, don't round or estimate

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
    
    # Enhanced search configuration for count queries and better coverage
    is_count_query = False
    if "how many" in (metadata_filter.get("query", "").lower() if isinstance(metadata_filter, dict) else ""):
        is_count_query = True
    
    # For count queries, increase retrieval to get more comprehensive results
    search_kwargs = {
        "k": 100 if is_count_query else 200,  # Larger pool for count queries
        "score_threshold": 0.3 if is_count_query else 0.5,  # Lower threshold for count queries
        "fetch_k": 150 if is_count_query else 300,  # Fetch more candidates for count queries
        "lambda_mult": 0.7,  # Balance between relevance (1.0) and diversity (0.0)
        "filter": metadata_filter if metadata_filter and len(metadata_filter) > 0 else None
    }
    
    # Use MMR (Maximum Marginal Relevance) search for more diverse results
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Use Maximum Marginal Relevance
        search_kwargs=search_kwargs
    )
    
    # Compression to extract the most relevant parts of documents
    #llm = Ollama(model="llama3.2")
    llm = Ollama(model="gemma3")
    # Debug: try a simple retrieval to verify it works
    if metadata_filter:
        try:
            test_docs = retriever.get_relevant_documents("test query")
            print(f"Retrieved {len(test_docs)} documents with filter")
            if test_docs:
                print(f"Sample timestamp: {test_docs[0].metadata.get('timestamp', 'No timestamp')}")
                # Print score information if available
                if hasattr(test_docs[0], 'score'):
                    print(f"Score of first document: {test_docs[0].score}")
        except Exception as e:
            print(f"Error testing retriever: {e}")
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt,
            "document_separator": "\n\n",  # Clear separation between log entries
        },
        return_source_documents=True,  # Return source documents for verification
    )
