# === Directory Structure ===
# ai_log_monitoring_app/
# ├── main.py                  # Streamlit entrypoint
# ├── log_processor.py         # Splits logs based on patterns
# ├── vector_store.py          # Embedding + Chroma setup
# ├── rag_chain.py             # RAG chain setup
# └── application_logs.txt     # Sample log file



# === main.py ===
import streamlit as st

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(page_title="AI Log Monitoring", layout="wide", page_icon="📊")

from rag_chain import get_qa_chain
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from datetime import datetime, timedelta
import re
import os
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import watchdog.observers
import watchdog.events
import threading
from log_processor import process_logs
import json

# Add JavaScript to maintain scroll position
st.markdown("""
<script>
// Function to scroll to stored position or to the chat input
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have a stored position
    const scrollPos = sessionStorage.getItem('streamlitScrollPosition');
    
    // Function to scroll to chat input area
    function scrollToChatInput() {
        // Find the chat input element
        const chatInput = document.querySelector('.stChatInput');
        if(chatInput) {
            // Create an observer to wait for full page load
            const observer = new MutationObserver((mutations, obs) => {
                const chatInputArea = document.querySelector('.stChatInput');
                if(chatInputArea) {
                    chatInputArea.scrollIntoView({behavior: 'smooth', block: 'center'});
                    obs.disconnect(); // Stop observing once scrolled
                }
            });
            
            // Start observing document for chat input to be fully rendered
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }
    }
    
    // Wait a bit longer for Streamlit to fully render the UI
    setTimeout(function() {
        if(scrollPos) {
            window.scrollTo(0, parseInt(scrollPos));
        } else {
            // If no saved position, scroll to chat input
            scrollToChatInput();
        }
    }, 800);
});

// Save scroll position before form submission
window.addEventListener('beforeunload', function() {
    sessionStorage.setItem('streamlitScrollPosition', window.scrollY);
});

// Try to detect when a new message is added and scroll to chat input
const targetNode = document.body;
const config = { childList: true, subtree: true };
const callback = function(mutationsList, observer) {
    for(const mutation of mutationsList) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            // Check if a chat message was added
            if(document.querySelector('.stChatMessage:last-child')) {
                const chatInput = document.querySelector('.stChatInput');
                if(chatInput) {
                    chatInput.scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            }
        }
    }
};

// Create and start observer
setTimeout(() => {
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
}, 1000);
</script>
""", unsafe_allow_html=True)

# Add an HTML anchor for the chat input area
st.markdown('<div id="chat-anchor"></div>', unsafe_allow_html=True)

# File to save chat history
HISTORY_FILE = "chat_history.pkl"

# Check if history file exists and remove it to clear history on restart
if os.path.exists(HISTORY_FILE):
    try:
        os.remove(HISTORY_FILE)
        print(f"Removed chat history file on application restart")
    except Exception as e:
        print(f"Error removing chat history file: {e}")

# Initialize chat history in session state with an empty list (fresh start on every restart)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize real-time monitoring flag
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
    
# Initialize log file to monitor
if 'log_file_to_monitor' not in st.session_state:
    st.session_state.log_file_to_monitor = "application_logs.txt"

# Initialize monitor thread
if 'monitor_thread' not in st.session_state:
    st.session_state.monitor_thread = None

# Initialize log watcher observer
if 'log_observer' not in st.session_state:
    st.session_state.log_observer = None
    
# Initialize new logs counter
if 'new_logs_count' not in st.session_state:
    st.session_state.new_logs_count = 0
    
# Initialize alerts
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Initialize recent errors
if 'recent_errors' not in st.session_state:
    st.session_state.recent_errors = []
    
# Function to save chat history to file
def save_chat_history():
    try:
        with open(HISTORY_FILE, 'wb') as f:
            pickle.dump(st.session_state.chat_history, f)
        print(f"Saved {len(st.session_state.chat_history)} messages to history file")
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Class for monitoring log files for changes
class LogFileHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_position = 0
        
    def on_modified(self, event):
        if not event.is_directory:
            self.callback(event.src_path)

# Function to process new log entries
def process_new_log_entries(log_file_path):
    try:
        # Get current file size
        file_size = os.path.getsize(log_file_path)
        
        # If we have a last position saved
        if hasattr(st.session_state, 'last_log_position'):
            # If file size is smaller than last position, file was truncated or replaced
            if file_size < st.session_state.last_log_position:
                st.session_state.last_log_position = 0
                
            # If file size is the same, no changes
            if file_size == st.session_state.last_log_position:
                return
                
            # Read only new content
            with open(log_file_path, 'r') as f:
                f.seek(st.session_state.last_log_position)
                new_content = f.read()
                
            # Process new content if there is any
            if new_content.strip():
                # Write new content to a temporary file
                temp_file = 'temp_new_logs.txt'
                with open(temp_file, 'w') as f:
                    f.write(new_content)
                    
                # Process the new logs
                process_logs(temp_file, "./chroma_logsN")
                
                # Increment the new logs counter
                st.session_state.new_logs_count += new_content.count('\n')
                
                # Check for error logs and add to alerts if found
                error_pattern = r'(ERROR|EXCEPTION|CRITICAL|FATAL)'
                if re.search(error_pattern, new_content, re.IGNORECASE):
                    # Extract error lines
                    for line in new_content.splitlines():
                        if re.search(error_pattern, line, re.IGNORECASE):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.alerts.append({
                                'timestamp': timestamp,
                                'message': line.strip(),
                                'type': 'error'
                            })
                            st.session_state.recent_errors.append({
                                'timestamp': timestamp,
                                'message': line.strip()
                            })
                
                # Clean up
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
            # Update the last position
            st.session_state.last_log_position = file_size
        else:
            # First run, just save the current position
            st.session_state.last_log_position = file_size
    except Exception as e:
        print(f"Error processing new log entries: {e}")

# Function to start log monitoring
def start_log_monitoring(log_file_path):
    if st.session_state.monitoring_active:
        return  # Already monitoring
        
    try:
        # Initialize the last position
        st.session_state.last_log_position = os.path.getsize(log_file_path)
        
        # Create observer and handler
        event_handler = LogFileHandler(process_new_log_entries)
        observer = watchdog.observers.Observer()
        observer.schedule(event_handler, path=os.path.dirname(log_file_path) or '.', 
                         recursive=False)
        
        # Start the observer
        observer.start()
        
        # Store the observer
        st.session_state.log_observer = observer
        
        # Set monitoring as active
        st.session_state.monitoring_active = True
        st.session_state.log_file_to_monitor = log_file_path
        
        # Reset counters
        st.session_state.new_logs_count = 0
        
        print(f"Started monitoring log file: {log_file_path}")
    except Exception as e:
        print(f"Error starting log monitoring: {e}")

# Function to stop log monitoring
def stop_log_monitoring():
    if not st.session_state.monitoring_active:
        return  # Not monitoring
        
    try:
        # Stop the observer
        if st.session_state.log_observer:
            st.session_state.log_observer.stop()
            st.session_state.log_observer.join()
            st.session_state.log_observer = None
            
        # Set monitoring as inactive
        st.session_state.monitoring_active = False
        
        print("Stopped log monitoring")
    except Exception as e:
        print(f"Error stopping log monitoring: {e}")

# Function to analyze log files by time period
def analyze_logs_by_time_period(period="daily"):
    try:
        # Get all logs from vector store
        embedding = OllamaEmbeddings(model="mxbai-embed-large")
        vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
        results = vectordb.get()
        
        if not results or 'metadatas' not in results:
            return {}
        
        # Extract timestamps and convert to datetime objects
        timestamps = []
        for metadata in results['metadatas']:
            if 'timestamp' in metadata and metadata['timestamp']:
                try:
                    ts = datetime.strptime(metadata['timestamp'], "%Y-%m-%d %H:%M:%S,%f")
                    timestamps.append(ts)
                except ValueError:
                    try:
                        # Try alternative format
                        ts = datetime.strptime(metadata['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                        timestamps.append(ts)
                    except ValueError:
                        # Try without milliseconds
                        try:
                            ts = datetime.strptime(metadata['timestamp'], "%Y-%m-%d %H:%M:%S")
                            timestamps.append(ts)
                        except:
                            pass
        
        if not timestamps:
            return {}
            
        # Group by period
        counts = {}
        
        if period == "hourly":
            # Group by hour
            hour_counts = defaultdict(int)
            for ts in timestamps:
                hour_key = ts.strftime("%Y-%m-%d %H:00")
                hour_counts[hour_key] += 1
            counts = dict(hour_counts)
        
        elif period == "daily":
            # Group by day
            day_counts = defaultdict(int)
            for ts in timestamps:
                day_key = ts.strftime("%Y-%m-%d")
                day_counts[day_key] += 1
            counts = dict(day_counts)
            
        elif period == "weekly":
            # Group by ISO week
            week_counts = defaultdict(int)
            for ts in timestamps:
                week_key = f"{ts.year}-W{ts.isocalendar()[1]:02d}"
                week_counts[week_key] += 1
            counts = dict(week_counts)
            
        elif period == "monthly":
            # Group by month
            month_counts = defaultdict(int)
            for ts in timestamps:
                month_key = ts.strftime("%Y-%m")
                month_counts[month_key] += 1
            counts = dict(month_counts)
        
        return counts
    except Exception as e:
        print(f"Error analyzing logs by time period: {e}")
        return {}

# Function to analyze logs by level
def analyze_logs_by_level():
    try:
        # Get all logs from vector store
        embedding = OllamaEmbeddings(model="mxbai-embed-large")
        vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
        results = vectordb.get()
        
        if not results or 'metadatas' not in results:
            return {}
        
        # Count by level
        level_counts = defaultdict(int)
        for metadata in results['metadatas']:
            if 'level' in metadata and metadata['level']:
                level = metadata['level']
                level_counts[level] += 1
        
        return dict(level_counts)
    except Exception as e:
        print(f"Error analyzing logs by level: {e}")
        return {}

# Function to analyze logs by component
def analyze_logs_by_component():
    try:
        # Get all logs from vector store
        embedding = OllamaEmbeddings(model="mxbai-embed-large")
        vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
        results = vectordb.get()
        
        if not results or 'metadatas' not in results:
            return {}
        
        # Count by component
        component_counts = defaultdict(int)
        for metadata in results['metadatas']:
            if 'source' in metadata and metadata['source']:
                component = metadata['source']
                component_counts[component] += 1
            elif 'component' in metadata and metadata['component']:
                component = metadata['component']
                component_counts[component] += 1
        
        return dict(component_counts)
    except Exception as e:
        print(f"Error analyzing logs by component: {e}")
        return {}

# Function to build metadata filter for time-based queries
def build_metadata_filter(query: str):
    now = datetime.now()
    
    # For time-based queries, use timestamp_unix which is a numeric field
    # that can be properly filtered with $gte and $lte operators
    print(f"build_metadata_filter: {query}")
    if "last 24 hours" in query.lower():
        cutoff = now - timedelta(hours=24)
        print("inside last 24 hours filter")
        # Convert to Unix timestamp (seconds since epoch)
        cutoff_unix = cutoff.timestamp()
        # Only return documents with timestamps greater than or equal to cutoff
        return {"timestamp_unix": {"$gte": cutoff_unix}}

    elif "last week" in query.lower():
        print("inside last week filter")
        # Calculate the start and end of the previous calendar week
        # Get the current date
        today = now.date()
        # Calculate the start of this week (Monday)
        start_of_this_week = today - timedelta(days=today.weekday())
        # Calculate the end of last week (Sunday)
        end_of_last_week = start_of_this_week - timedelta(days=1)
        # Calculate the start of last week (Monday)
        start_of_last_week = end_of_last_week - timedelta(days=6)
        
        # Convert to datetime with time component
        start_datetime = datetime.combine(start_of_last_week, datetime.min.time())
        end_datetime = datetime.combine(end_of_last_week, datetime.max.time())
        print(f"start_datetime: {start_datetime}, end_datetime: {end_datetime}");
        # Convert to Unix timestamps
        start_unix = start_datetime.timestamp()
        end_unix = end_datetime.timestamp()
        
        # Return filter for timestamps between start and end of last week
        return {"$and": [
            {"timestamp_unix": {"$gte": start_unix}},
            {"timestamp_unix": {"$lte": end_unix}}
        ]}
        
    elif "last month" in query.lower():
            print("Detected last month filter")
            cutoff = now - timedelta(days=30)
            # Convert to Unix timestamp (seconds since epoch)
            cutoff_unix = cutoff.timestamp()
            # Only return documents with timestamps greater than or equal to cutoff
            return {"timestamp_unix": {"$gte": cutoff_unix}}
        
        # For error/warning filters, look at the level field
    elif "error" in query.lower():
            return {"level": "ERROR"}
    elif "warning" in query.lower():
            return {"level": "WARNING"}
    elif "info" in query.lower():
            return {"level": "INFO"}
    elif "debug" in query.lower():
            return {"level": "DEBUG"}
        
        # No specific time or level filter found
    return None

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

def count_logs_by_level(log_level: str):
    """
    Directly count logs by level from the vector store without relying on LLM interpretation.
    This ensures accurate counts for ERROR, WARN, INFO, etc. log types.
    Special handling for ERROR/EXCEPTION: treats them as the same category.
    """
    try:
        embedding = OllamaEmbeddings(model="mxbai-embed-large")
        vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
        
        # Special case: treat ERROR and EXCEPTION as the same category
        if log_level.upper() in ["ERROR", "EXCEPTION"]:
            # Query using OR to get both ERROR and EXCEPTION logs
            filter_dict = {"$or": [{"level": "ERROR"}, {"level": "EXCEPTION"}]}
            display_level = "ERROR/EXCEPTION"
        else:
            # Query using the level metadata field which was stored during log processing
            filter_dict = {"level": log_level.upper()}
            display_level = log_level.upper()
        
        results = vectordb.get(where=filter_dict)
        
        if results and 'ids' in results:
            count = len(results['ids'])
            
            # Return both count and sample logs
            sample_logs = []
            if 'documents' in results and count > 0:
                # Limit to 5 examples
                for i in range(min(5, count)):
                    sample_logs.append(results['documents'][i])
                    
            return {
                "count": count,
                "examples": sample_logs,
                "display_level": display_level
            }
        else:
            return {"count": 0, "examples": [], "display_level": display_level}
            
    except Exception as e:
        print(f"Error counting logs: {e}")
        return {"count": 0, "examples": [], "error": str(e), "display_level": log_level.upper()}

def count_logs_by_topic_and_level(log_level: str, topic: str = None):
    """
    Count logs by both level and topic, allowing for filtering by content (like "database")
    This allows for finding specific types of errors, like "database errors" only.
    """
    try:
        embedding = OllamaEmbeddings(model="mxbai-embed-large")
        vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
        
        # Start with level filter
        if log_level.upper() in ["ERROR", "EXCEPTION"]:
            filter_dict = {"$or": [{"level": "ERROR"}, {"level": "EXCEPTION"}]}
            display_level = "ERROR/EXCEPTION"
        else:
            filter_dict = {"level": log_level.upper()}
            display_level = log_level.upper()
        
        # Get all logs of the specified level first
        level_results = vectordb.get(where=filter_dict)
        
        if not level_results or 'ids' not in level_results or not level_results['ids']:
            return {"count": 0, "examples": [], "display_level": display_level, "topic": topic}
        
        # If topic is specified, filter by content
        filtered_indices = []
        if topic and 'documents' in level_results:
            for i, doc in enumerate(level_results['documents']):
                # Check if the document contains the topic keyword
                if topic.lower() in doc.lower():
                    filtered_indices.append(i)
        else:
            # If no topic filter, use all results
            filtered_indices = list(range(len(level_results['ids'])))
        
        # Extract the filtered documents
        filtered_docs = [level_results['documents'][i] for i in filtered_indices] if filtered_indices else []
        count = len(filtered_docs)
        
        # Format display topic
        display_topic = topic.capitalize() if topic else None
        
        return {
            "count": count,
            "examples": filtered_docs[:5],  # Limit to 5 examples
            "display_level": display_level,
            "topic": display_topic
        }
            
    except Exception as e:
        print(f"Error counting logs: {e}")
        return {"count": 0, "examples": [], "error": str(e), "display_level": log_level.upper(), "topic": topic}

def extract_month_from_query(query: str):
    """Extract month information from a query string."""
    # Check for month mentions in the query
    month_pattern = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(\d{4}))?"
    month_match = re.search(month_pattern, query.lower())
    print(f"Inside extract_month_from_query -- Month match: {month_match}")
    month_filter = None
    month_info = {}
    
    if month_match:
        month_str = month_match.group(1)
        year_str = month_match.group(2) if month_match.group(2) else str(datetime.now().year)
        month_names = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        short_month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        
        month_str_lower = month_str.lower()
        if month_str_lower.startswith(tuple(short_month_names)):
            # Find which month prefix matches
            for i, m in enumerate(short_month_names):
                if month_str_lower.startswith(m):
                    month_num = i + 1
                    month_name = month_names[i].capitalize()
                    break
        else:
            month_num = 1  # fallback
            month_name = "January"
        
        year = int(year_str)
        start_date = datetime(year, month_num, 1)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end_date = datetime(year, month_num + 1, 1) - timedelta(seconds=1)
            
        start_unix = start_date.timestamp()
        end_unix = end_date.timestamp()
        
        month_filter = {"$and": [
            {"timestamp_unix": {"$gte": start_unix}},
            {"timestamp_unix": {"$lte": end_unix}}
        ]}
        
        month_info = {
            "month_name": month_name,
            "month_num": month_num,
            "year": year,
            "start_date": start_date,
            "end_date": end_date
        }
        
    return month_filter, month_info

def extract_level_from_query(query: str):
    """Extract log level information from a query string."""
    # Common log levels
    log_levels = ["error", "warn", "warning", "info", "debug", "trace", "exception", "critical", "fatal"]
    
    # Check for explicit level mentions
    level_filter = None
    level_name = None
    
    query_lower = query.lower()
    
    # Special case: treat "exception" as "error"
    if "exception" in query_lower or "error" in query_lower:
        print("Detected ERROR/EXCEPTION in query")
        level_filter = {"$or": [{"level": "ERROR"}, {"level": "EXCEPTION"}]}
        level_name = "ERROR/EXCEPTION"
        return level_filter, level_name
    
    # Check for each log level in the query
    for level in log_levels:
        if level in query_lower:
            level_upper = level.upper()
            level_filter = {"level": level_upper}
            level_name = level_upper
            break
    
    return level_filter, level_name

def show_logs_from_query(query: str):
    """Process 'show logs' style queries and return formatted logs."""
    # Extract month filter if present
    month_filter, month_info = extract_month_from_query(query)
    
    # Extract level filter if present
    level_filter, level_name = extract_level_from_query(query)
    
    # Build the complete filter
    filter_dict = {}
    
    if month_filter and level_filter:
        # Handle special case for ERROR/EXCEPTION combined filter
        if level_name == "ERROR/EXCEPTION":
            filter_dict = {"$and": [month_filter["$and"][0], month_filter["$and"][1], 
                                  {"$or": [{"level": "ERROR"}, {"level": "EXCEPTION"}]}]}
        else:
            # Both month and level specified
            filter_dict = {"$and": [{"level": level_name}, 
                                  month_filter["$and"][0], 
                                  month_filter["$and"][1]]}
    elif month_filter:
        # Only month specified
        filter_dict = month_filter
    elif level_filter:
        # Only level specified
        filter_dict = level_filter
    
    # Query the vector database
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
    results = vectordb.get(where=filter_dict)
    
    # Format the results
    if results and 'ids' in results and results['ids']:
        count = len(results['ids'])
        
        # Create header text based on filters
        header_text = f"Showing {count} logs"
        if level_name:
            header_text += f" with level {level_name}"
        if month_info:
            header_text += f" from {month_info['month_name']} {month_info['year']}"
        header_text += ":"
        
        # Format logs with metadata
        logs_text = ""
        if 'documents' in results and 'metadatas' in results:
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
                timestamp = metadata.get('timestamp', 'Unknown time')
                level = metadata.get('level', 'Unknown level')
                source = metadata.get('source', 'Unknown source')
                
                logs_text += f"\n\n**Log {i}:**\n"
                logs_text += f"- **Time:** {timestamp}\n"
                logs_text += f"- **Level:** {level}\n"
                logs_text += f"- **Source:** {source}\n"
                logs_text += f"```\n{doc}\n```"
        
        # Full answer text
        answer_text = f"{header_text}\n{logs_text}"
        
        # If too many logs, add a note
        if count > 20:
            answer_text += f"\n\n_Note: Showing only the first 20 of {count} logs. Please refine your query if you need specific logs._"
            
        return answer_text
    else:
        # No logs found
        not_found_text = "No logs found"
        if level_name:
            not_found_text += f" with level {level_name}"
        if month_info:
            not_found_text += f" from {month_info['month_name']} {month_info['year']}"
        return not_found_text + "."

# Function to handle form submission and clear input
def submit_query():
    if st.session_state.query_input:
        # Store the query value before clearing
        query_text = st.session_state.query_input
        # Clear the input immediately
        st.session_state.query_input = ""
        # Process the query
        process_query(query_text)

# Function to process the query and update chat history
def process_query(query):
    # Add user query to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })
    
    # Check if this is a log count query that needs special handling
    count_patterns = [
        r"how many (\w+) logs",
        r"count (\w+) logs",
        r"number of (\w+) logs",
        r"total (\w+) logs",
    ]
    
    is_count_query = False
    log_level = None
    
    for pattern in count_patterns:
        match = re.search(pattern, query.lower())
        if match:
            log_level = match.group(1).upper()
            is_count_query = True
            break
    print(f"Detected count query: {is_count_query}, log level: {log_level}")
    # Special direct counting path for count queries
    if is_count_query and log_level:
        # Get direct count from vector store
        count_result = count_logs_by_level(log_level)
        
        # Format response with count and examples
        if count_result["count"] > 0:
            answer_text = f"There are exactly **{count_result['count']} {count_result['display_level']} logs**. Here are some examples:\n\n"
            
            for i, example in enumerate(count_result["examples"][:5]):
                answer_text += f"**Example {i+1}:**\n```\n{example}\n```\n\n"
                
            # Include note about accurate counting
           # answer_text += "_Note: This count is exact and was obtained by directly querying the database._"
        else:
            answer_text = f"There are no {count_result['display_level']} logs found in the database."
            
        # Add response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text
        })
        
    # Regular RAG processing path for non-count queries
    else:
        metadata_filter = build_metadata_filter(query)
        print(f"metadata_filter: {metadata_filter}")
        qa_chain = get_qa_chain(metadata_filter)
        
        # Prepare context from previous conversation if available
        context = ""
        if len(st.session_state.chat_history) > 1:
            # Get last exchanges for better context retention
            recent_history = st.session_state.chat_history[-20:]
            for chat in recent_history:
                if chat["role"] == "user":
                    context += f"User: {chat['content']}\n"
                else:
                    context += f"Assistant: {chat['content']}\n"
        
        # Add context to the query if there's history
        if context:
            enhanced_query = (
                f"Previous conversation context:\n{context}\n\n"
                f"Current question: {query}\n\n"
                f"Instructions:\n"
                f"1. Consider the previous conversation context when interpreting the current question\n"
                f"2. If the current question contains pronouns or references to previous messages, resolve them\n"
                f"3. Focus your search on log entries most relevant to the CURRENT question\n"
                f"4. Provide a complete and accurate answer based on the log data, not just the conversation history\n"
            )
        else:
            enhanced_query = query
        
        # Get answer from the RAG chain
        result = qa_chain.invoke(enhanced_query)
        
        # Extract answer and source documents
        if isinstance(result, dict):
            answer = result.get("result", "No result found")
            source_docs = result.get("source_documents", [])
        else:
            answer = result
            source_docs = []
    
        # Extract the text portion of the response
        answer_text = extract_text_from_response(answer)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text
        })
    
    # Save chat history
    save_chat_history()

# Function to handle text area clearing with callbacks
def clear_text_input():
    # This function gets called when we want to reset the input field
    # It sets the field's value to empty string
    st.session_state["query_input"] = ""

# Function to reset the input field - this must be defined before the session state is used
def reset_input():
    st.session_state.query_input = ""

# Create a special callback for processing queries
def handle_ask_button():
    # Make sure we have a query to process
    if st.session_state.query_input and st.session_state.query_input.strip():
        # Get the current input
        query_text = st.session_state.query_input
        # Process immediately (no pending flag needed)
        process_query(query_text)
        # Clear input after processing
        st.session_state.query_input = ""

# Set up the Streamlit page
st.title("Log Monitoring Assistant")

# Create tabs
tab1, tab2 = st.tabs(["Ask Questions", "Vector DB Viewer"])

# Tab 1: Ask Questions
with tab1:
    st.header("Log Monitoring Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_user_query" not in st.session_state:
        st.session_state.pending_user_query = None
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Step 1: User submits a question
    user_query = st.chat_input("Ask a question about your logs...")
    if user_query and not st.session_state.awaiting_response:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.pending_user_query = user_query
        st.session_state.awaiting_response = True
        st.rerun()

    # Step 2: If awaiting response, show spinner, process, and append answer
    if st.session_state.awaiting_response and st.session_state.pending_user_query:
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                query = st.session_state.pending_user_query
                
                # Check if this is a count query with topic filter
                # Check for specific error topic queries like "database errors" or "sql errors"
                database_error_pattern = r"(?:how many|count|number of|total)\s+(?:database|sql|db|query).*(?:errors|failed|failure|error)"
                db_error_match = re.search(database_error_pattern, query.lower())
                
                if db_error_match:
                    # This is a database error count query - use topic filtering
                    count_result = count_logs_by_topic_and_level("ERROR", "database")
                    
                    # Format response with count and examples
                    if count_result["count"] > 0:
                        answer_text = f"There are exactly **{count_result['count']} database {count_result['display_level']} logs** in the database. Here are some examples:\n\n"
                        
                        for i, example in enumerate(count_result["examples"][:5]):
                            answer_text += f"**Example {i+1}:**\n```\n{example}\n```\n\n"
                            
                        # Include note about accurate counting
                        answer_text += "_Note: This count is exact and includes only database-related errors._"
                    else:
                        answer_text = f"There are no database-related {count_result['display_level']} logs found."
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                    
                    # Reset state
                    st.session_state.awaiting_response = False
                    st.session_state.pending_user_query = None
                    
                    # Save and rerun
                    save_chat_history() 
                    st.rerun()
                
                # Check if this is a count query
                count_patterns = [
                    r"how many (\w+) logs",
                    r"count (\w+) logs",
                    r"number of (\w+) logs",
                    r"total (\w+) logs",
                ]
                is_count_query = False
                log_level = None
                for pattern in count_patterns:
                    match = re.search(pattern, query.lower())
                    if match:
                        log_level = match.group(1).upper()
                        is_count_query = True
                        break
                
                # Check if this is a "show logs" query
                show_patterns = [
                    r"show (?:all )?(logs|errors?|exceptions?|warnings?|infos?)",
                    r"display (?:all )?(logs|errors?|exceptions?|warnings?|infos?)",
                    r"list (?:all )?(logs|errors?|exceptions?|warnings?|infos?)",
                    r"get (?:all )?(logs|errors?|exceptions?|warnings?|infos?)",
                ]
                is_show_query = False
                for pattern in show_patterns:
                    if re.search(pattern, query.lower()):
                        is_show_query = True
                        break
                
                # Process the query based on its type
                if is_show_query:
                    # Use the show_logs_from_query function for "show logs" queries
                    answer_text = show_logs_from_query(query)
                elif is_count_query and log_level:
                    # --- Month filter logic ---
                    month_filter = None
                    month_pattern = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(\d{4}))?"
                    month_match = re.search(month_pattern, query.lower())
                    if month_match:
                        month_str = month_match.group(1)
                        year_str = month_match.group(2) if month_match.group(2) else str(datetime.now().year)
                        month_names = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                        short_month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
                        month_str = month_str[:3].lower()
                        if month_str in short_month_names:
                            month_num = short_month_names.index(month_str) + 1
                        else:
                            month_num = 1  # fallback, should not happen
                        year = int(year_str)
                        start_date = datetime(year, month_num, 1)
                        if month_num == 12:
                            end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
                        else:
                            end_date = datetime(year, month_num + 1, 1) - timedelta(seconds=1)
                        start_unix = start_date.timestamp()
                        end_unix = end_date.timestamp()
                        month_filter = {"$and": [
                            {"timestamp_unix": {"$gte": start_unix}},
                            {"timestamp_unix": {"$lte": end_unix}}
                        ]}
                    
                    # Build filter for vector DB
                    # Special handling for ERROR/EXCEPTION
                    display_level = log_level
                    if log_level.upper() in ["ERROR", "EXCEPTION"]:
                        # Use OR filter to get both ERROR and EXCEPTION logs
                        level_filter = {"$or": [{"level": "ERROR"}, {"level": "EXCEPTION"}]}
                        display_level = "ERROR/EXCEPTION"
                    else:
                        # Normal level filtering
                        level_filter = {"level": log_level}
                    
                    # Apply month filter if present
                    if month_filter:
                        # Combine month filter with level filter
                        if log_level.upper() in ["ERROR", "EXCEPTION"]:
                            # Special case for ERROR/EXCEPTION with month filter
                            filter_dict = {"$and": [
                                {"$or": [{"level": "ERROR"}, {"level": "EXCEPTION"}]},
                                month_filter["$and"][0],
                                month_filter["$and"][1]
                            ]}
                        else:
                            # Normal level with month filter
                            filter_dict = {"$and": [
                                {"level": log_level},
                                month_filter["$and"][0],
                                month_filter["$and"][1]
                            ]}
                    else:
                        # Only level filter, no month filter
                        filter_dict = level_filter
                    
                    # Query the database
                    embedding = OllamaEmbeddings(model="mxbai-embed-large")
                    vectordb = Chroma(persist_directory="./chroma_logsN", embedding_function=embedding)
                    results = vectordb.get(where=filter_dict)
                    count = len(results['ids']) if results and 'ids' in results else 0
                    
                    # Generate response
                    month_text = f" in {month_str.capitalize()} {year}" if month_filter else ""
                    if count > 0:
                        answer_text = f"There are exactly **{count} {display_level} logs{month_text}** in the database. Here are some examples:\n\n"
                        if 'documents' in results:
                            for i, example in enumerate(results['documents'][:5]):
                                answer_text += f"**Example {i+1}:**\n```\n{example}\n```\n\n"
                        answer_text += "_Note: This count is exact and was obtained by directly querying the database._"
                    else:
                        answer_text = f"There are no {display_level} logs found{month_text}."
                else:
                    # Generic question - use RAG chain
                    metadata_filter = build_metadata_filter(query)
                    print(f"metadata_filter: {metadata_filter}")
                    qa_chain = get_qa_chain(metadata_filter)
                    context = ""
                    if len(st.session_state.chat_history) > 1:
                        recent_history = st.session_state.chat_history[-10:]
                        for chat in recent_history:
                            if chat["role"] == "user":
                                context += f"User: {chat['content']}\n"
                            elif chat["role"] == "assistant":
                                context += f"Assistant: {chat['content']}\n"
                    if context:
                        enhanced_query = (
                            f"Previous conversation context:\n{context}\n\n"
                            f"Current question: {query}\n\n"
                            f"Instructions:\n"
                            f"1. Consider the previous conversation context when interpreting the current question\n"
                            f"2. If the current question contains pronouns or references to previous messages, resolve them\n"
                            f"3. Focus your search on log entries most relevant to the CURRENT question\n"
                            f"4. Provide a complete and accurate answer based on the log data\n"
                        )
                    else:
                        enhanced_query = query
                    result = qa_chain.invoke(enhanced_query)
                    answer_text = extract_text_from_response(result)
                
                st.markdown(answer_text)
                st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
                st.session_state.pending_user_query = None
                st.session_state.awaiting_response = False
                save_chat_history()
                st.rerun()

# Tab 2: Vector DB Viewer
with tab2:
    st.markdown("### Vector Database Contents")
    
    # Add a refresh button with better styling
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh Data", key="refresh_db_button", use_container_width=True):
            with st.spinner("Refreshing database data..."):
                st.session_state.vector_data = get_vector_db_data()
                st.success("Data refreshed successfully!")
    
    # Add some spacing
    st.markdown("---")
    
    # Initialize vector_data in session_state if it doesn't exist
    if 'vector_data' not in st.session_state:
        with st.spinner("Loading vector database data..."):
            st.session_state.vector_data = get_vector_db_data()
    
    # Display the data with better formatting
    if st.session_state.vector_data:
        # Create a simplified DataFrame for display
        df_display = pd.DataFrame([{
            'ID': item['ID'], 
            'Content Preview': item['Content'],
            'Source': item['Metadata'].get('source', 'Unknown')
        } for item in st.session_state.vector_data])
        
        # Display the dataframe with better styling
        st.dataframe(df_display, use_container_width=True, height=300)
        
        # Document viewer with better styling
        st.markdown("### Document Details")
        selected_id = st.selectbox("Select document ID to view details:", 
                                 [item['ID'] for item in st.session_state.vector_data])
        
        if selected_id:
            selected_doc = next((item for item in st.session_state.vector_data if item['ID'] == selected_id), None)
            if selected_doc:
                tabs = st.tabs(["Content", "Metadata"])
                with tabs[0]:
                    st.text_area("Full Content", selected_doc['Full Content'], height=250)
                with tabs[1]:
                    st.json(selected_doc['Metadata'])
    else:
        st.warning("No data found in the vector database.")
        
    def build_metadata_filter2(query: str):
        """Build metadata filter based on query."""
        now = datetime.now()
        
        if "last month" in query.lower():
            cutoff = now - timedelta(days=30)
            # Convert to Unix timestamp (seconds since epoch)
            cutoff_unix = cutoff.timestamp()
            # Only return documents with timestamps greater than or equal to cutoff
            return {"timestamp_unix": {"$gte": cutoff_unix}}
        
        # For error/warning filters, look at the level field
        if "error" in query.lower():
            return {"level": "ERROR"}
        elif "warning" in query.lower():
            return {"level": "WARNING"}
        elif "info" in query.lower():
            return {"level": "INFO"}
        elif "debug" in query.lower():
            return {"level": "DEBUG"}
        
        # No specific time or level filter found
        return None

if __name__ == "__main__":
    # Everything is handled by Streamlit's UI system
    pass
