# === Directory Structure ===
# ai_log_monitoring_app/
# ├── main.py                  # Streamlit entrypoint
# ├── log_processor.py         # Splits logs based on patterns
# ├── vector_store.py          # Embedding + Chroma setup
# ├── rag_chain.py             # RAG chain setup
# └── application_logs.txt     # Sample log file



# === main.py ===
import streamlit as st
from rag_chain import get_qa_chain
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# main.py (add this at the top)
from datetime import datetime, timedelta
import re

def build_metadata_filter(query: str):
    now = datetime.now()
    
    # For time-based queries, use timestamp_unix which is a numeric field
    # that can be properly filtered with $gte and $lte operators
    
    if "last 24 hours" in query.lower():
        cutoff = now - timedelta(hours=24)
        # Convert to Unix timestamp (seconds since epoch)
        cutoff_unix = cutoff.timestamp()
        # Only return documents with timestamps greater than or equal to cutoff
        return {"timestamp_unix": {"$gte": cutoff_unix}}

    elif "last week" in query.lower():
        cutoff = now - timedelta(days=8)
        # Convert to Unix timestamp (seconds since epoch)
        cutoff_unix = cutoff.timestamp()
        # Only return documents with timestamps greater than or equal to cutoff
        return {"timestamp_unix": {"$gte": cutoff_unix}}

    # Handle specific month queries like "January logs" or "logs from March"
    month_pattern = r"(?:logs from|logs in|for|from|in)\s+(?:the month of\s+)?(\w+)(?:\s+(\d{4}))?"
    month_match = re.search(month_pattern, query.lower())
    if month_match:
        month_name = month_match.group(1).capitalize()
        year_str = month_match.group(2) if month_match.group(2) else str(now.year)
        
        try:
            # Convert month name to number
            month_names = ["January", "February", "March", "April", "May", "June", 
                          "July", "August", "September", "October", "November", "December"]
            # Also handle short month names
            short_month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            # Find the month number (1-12)
            if month_name in month_names:
                month_num = month_names.index(month_name) + 1
            elif month_name.capitalize() in month_names:
                month_num = month_names.index(month_name.capitalize()) + 1
            elif month_name in short_month_names:
                month_num = short_month_names.index(month_name) + 1
            elif month_name.capitalize() in short_month_names:
                month_num = short_month_names.index(month_name.capitalize()) + 1
            else:
                # Try partial matching for month names
                for i, m in enumerate(month_names):
                    if m.lower().startswith(month_name.lower()):
                        month_num = i + 1
                        break
                else:
                    print(f"Could not match month name: {month_name}")
                    return {}
            
            year = int(year_str)
            
            # Create start and end dates for the month
            start_date = datetime(year, month_num, 1)  # First day of month
            
            # Last day of month (using the fact that going to day 1 of next month - 1 day gives last day)
            if month_num == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
            
            # Include the entire day for the end date
            end_date = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59)
            
            # Convert to Unix timestamps
            start_unix = start_date.timestamp()
            end_unix = end_date.timestamp()
            
            print(f"Filtering logs for {month_name} {year} - from {start_date.isoformat()} to {end_date.isoformat()}")
            
            # Use $and to combine two separate filters
            return {
                "$and": [
                    {"timestamp_unix": {"$gte": start_unix}},
                    {"timestamp_unix": {"$lte": end_unix}}
                ]
            }
        except Exception as e:
            print(f"Error parsing month: {e}")
            return {}

    # Handle formats like "from Jan 10 to Jan 12"
    match = re.search(r"from (\w+ \d{1,2}) to (\w+ \d{1,2})", query, re.IGNORECASE)
    if match:
        start_str, end_str = match.groups()
        try:
            this_year = now.year
            start_date = datetime.strptime(f"{start_str} {this_year}", "%b %d %Y")
            end_date = datetime.strptime(f"{end_str} {this_year} 23:59:59", "%b %d %Y %H:%M:%S")
            
            # Convert to Unix timestamps
            start_unix = start_date.timestamp()
            end_unix = end_date.timestamp()
            
            # Use $and to combine two separate filters instead of two operators on one field
            return {
                "$and": [
                    {"timestamp_unix": {"$gte": start_unix}},
                    {"timestamp_unix": {"$lte": end_unix}}
                ]
            }
        except Exception as e:
            print(f"Error parsing date range: {e}")
            pass

    return {}

# Initialize the RAG QA chain
qa_chain = get_qa_chain()

# Function to get vector database data
def get_vector_db_data():
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
    
    # Get all documents from the vector store
    results = vectordb.get()
    
    # Create a list to store document data
    documents_data = []
    
    # Process the results
    if results and 'documents' in results and 'metadatas' in results and 'ids' in results:
        for i, (doc, metadata, doc_id) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
            # Create a dictionary for each document
            doc_data = {
                'ID': doc_id,
                'Content': doc[:100] + "..." if len(doc) > 100 else doc,  # Truncate long content
                'Full Content': doc,
                'Metadata': metadata
            }
            documents_data.append(doc_data)
    
    return documents_data

# Extract text from response
def extract_text_from_response(response):
    # If response is a dictionary and has a 'result' key, extract that
    if isinstance(response, dict) and 'result' in response:
        return response['result']
    # If response is already a string, return it directly
    elif isinstance(response, str):
        return response
    # If it's another type, convert to string
    else:
        return str(response)

# Set up the Streamlit page
st.set_page_config(page_title="Log Monitor", layout="wide")
st.title("Log Monitoring Assistant")

# Create tabs
tab1, tab2 = st.tabs(["Ask Questions", "Vector DB Viewer"])

# Tab 1: Ask Questions
with tab1:
    # Create the query input area
    query = st.text_area("Ask a log question...", height=150)

    # Create the submit button
    if st.button("Ask"):
        if query:
            # Display a spinner while processing
            with st.spinner("Processing your query..."):
                metadata_filter = build_metadata_filter(query)
                print(f"metadata_filter: {metadata_filter}")
                qa_chain = get_qa_chain(metadata_filter)
                # Get answer from the RAG chain using invoke instead of run
                result = qa_chain.invoke(query)
                
                # Extract answer and source documents
                if isinstance(result, dict):
                    answer = result.get("result", "No result found")
                    source_docs = result.get("source_documents", [])
                else:
                    answer = result
                    source_docs = []
                
            # Display results
            st.subheader("Question:")
            st.write(query)
            st.subheader("Answer:")
            # Extract and display the text portion of the response
            answer_text = extract_text_from_response(answer)
            st.markdown(answer_text)
            
            # Display source documents for verification
            if source_docs:
                with st.expander("View Source Log Entries", expanded=False):
                    #st.write("These are the actual log entries used to generate the answer:")
                    print("These are the actual log entries used to generate the answer:")
                    for i, doc in enumerate(source_docs):
                        #st.markdown(f"**Log Entry {i+1}:**")
                        #st.code(doc.page_content, language="text")
                        print(f"**Log Entry {i+1}:**")
                        print(doc.page_content)
            else:
                st.info("No source log entries were retrieved for this query.")

# Tab 2: Vector DB Viewer
with tab2:
    st.header("Vector Database Contents")
    
    # Add a refresh button
    if st.button("Refresh Vector DB Data"):
        st.session_state.vector_data = get_vector_db_data()
    
    # Initialize vector_data in session_state if it doesn't exist
    if 'vector_data' not in st.session_state:
        with st.spinner("Loading vector database data..."):
            st.session_state.vector_data = get_vector_db_data()
    
    # Display the data
    if st.session_state.vector_data:
        # Create a simplified DataFrame for display
        df_display = pd.DataFrame([{
            'ID': item['ID'], 
            'Content Preview': item['Content'],
            'Source': item['Metadata'].get('source', 'Unknown')
        } for item in st.session_state.vector_data])
        
        st.dataframe(df_display, use_container_width=True)
        
        # Document viewer
        st.subheader("Document Details")
        selected_id = st.selectbox("Select document ID to view details:", 
                                 [item['ID'] for item in st.session_state.vector_data])
        
        if selected_id:
            selected_doc = next((item for item in st.session_state.vector_data if item['ID'] == selected_id), None)
            if selected_doc:
                st.text_area("Full Content", selected_doc['Full Content'], height=300)
                st.json(selected_doc['Metadata'])
    else:
        st.warning("No data found in the vector database.")
