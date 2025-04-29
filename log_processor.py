import re
from langchain.schema import Document
from datetime import datetime
from typing import Dict, List, Optional, Pattern, Callable, Tuple

# Define log format registry
LOG_FORMATS = {
    "standard": {
        "name": "Standard Format",
        "description": "YYYY-MM-DD HH:MM:SS [LEVEL] [Component] Message",
        "regex": re.compile(r"""
            ^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})          # Timestamp
            \s+
            (?:\[(?P<level1>[A-Z]+)\]|(?P<level2>[A-Z]+))                 # [LEVEL] or LEVEL
            (?:\s+\[(?P<component>[^\]]+)\])?                             # Optional [Component]
            \s+(?P<message>[^\n]+)                                        # First line of message
            (?P<exception>(?:\n\s+(?!\d{4}-\d{2}-\d{2}).*)*)               # Multi-line stack trace
            """, re.VERBOSE | re.MULTILINE),
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "entry_split_pattern": r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    },
    "apache": {
        "name": "Apache Log Format",
        "description": "IP - - [DD/Mon/YYYY:HH:MM:SS +ZZZZ] \"METHOD URL HTTP/1.x\" STATUS SIZE",
        "regex": re.compile(r"""
            ^(?P<ip>\S+)\s+-\s+-\s+                              # IP address
            \[(?P<timestamp>\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4})\]\s+ # Timestamp
            "(?P<method>[A-Z]+)\s+(?P<url>\S+)\s+HTTP/[\d\.]+"\s+ # HTTP request
            (?P<status>\d+)\s+                                   # Status code
            (?P<size>\d+|-)\s*                                   # Size
            (?P<message>.*)$                                     # Optional message
            """, re.VERBOSE | re.MULTILINE),
        "timestamp_format": "%d/%b/%Y:%H:%M:%S %z",
        "entry_split_pattern": r"(?=\S+ - - \[\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}\])"
    },
    "json": {
        "name": "JSON Log Format",
        "description": "JSON structured logs with timestamp, level, and message fields",
        "regex": None,  # Special handling for JSON logs
        "timestamp_format": None,  # Depends on the JSON format
        "entry_split_pattern": r"(?=\{)"
    },
    "syslog": {
        "name": "Syslog Format",
        "description": "MMM DD HH:MM:SS hostname process[pid]: message",
        "regex": re.compile(r"""
            ^(?P<timestamp>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+ # Timestamp
            (?P<hostname>\S+)\s+                                           # Hostname
            (?P<process>\S+)(?:\[(?P<pid>\d+)\])?:                         # Process[pid]
            \s+(?P<message>.*)$                                            # Message
            """, re.VERBOSE | re.MULTILINE),
        "timestamp_format": "%b %d %H:%M:%S",
        "entry_split_pattern": r"(?=[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
    }
}

def detect_log_format(content: str) -> str:
    """Auto-detect the log format based on content sample"""
    # Create a small sample of the content for performance
    sample = content[:5000] if len(content) > 5000 else content
    
    format_scores = {}
    
    # Try each format and score by match count
    for format_id, format_info in LOG_FORMATS.items():
        if format_id == "json":
            # Special case for JSON
            if sample.strip().startswith("{") and "}" in sample:
                try:
                    import json
                    # Try to parse as JSON
                    lines = sample.strip().split("\n")
                    valid_json = 0
                    for line in lines[:10]:  # Check first 10 lines
                        try:
                            json.loads(line.strip())
                            valid_json += 1
                        except:
                            pass
                    format_scores[format_id] = valid_json
                except:
                    format_scores[format_id] = 0
        else:
            # For regex-based formats
            regex = format_info["regex"]
            if regex:
                entries = re.split(format_info["entry_split_pattern"], sample)
                matched = sum(1 for entry in entries if regex.match(entry.strip()))
                format_scores[format_id] = matched
    
    # Return the format with the highest score
    if format_scores:
        best_format = max(format_scores.items(), key=lambda x: x[1])
        if best_format[1] > 0:
            return best_format[0]
    
    # Default to standard if no good match
    return "standard"

def process_standard_log(entry: str, format_info: Dict) -> Optional[Dict]:
    """Process a log entry with standard format"""
    match = format_info["regex"].match(entry.strip())
    if not match:
        return None
        
    data = match.groupdict()
    try:
        parsed_time = datetime.strptime(data["timestamp"], format_info["timestamp_format"])
    except Exception:
        return None
        
    level = match.group("level1") or match.group("level2") or "UNKNOWN"
    message = match.group("message").strip()
    exception = match.group("exception").strip() if match.group("exception") else ""
    full_content = f"{message}\n{exception}" if exception else message
    timestamp = match.group("timestamp")
    
    return {
        "timestamp": timestamp,
        "timestamp_iso": parsed_time.isoformat(),
        "timestamp_unix": parsed_time.timestamp(),
        "month": parsed_time.strftime("%B"),
        "year": parsed_time.year,
        "day": parsed_time.day,
        "level": level,
        "component": data.get("component", "") or "Unknown",
        "has_exception": bool(exception),
        "message_preview": message[:50] if message else "",
        "message": message,
        "exception": exception,
        "raw": entry.strip()
    }

def process_apache_log(entry: str, format_info: Dict) -> Optional[Dict]:
    """Process an Apache format log entry"""
    match = format_info["regex"].match(entry.strip())
    if not match:
        return None
        
    data = match.groupdict()
    try:
        parsed_time = datetime.strptime(data["timestamp"], format_info["timestamp_format"])
    except Exception:
        return None
        
    status_code = data.get("status", "")
    # Determine level based on status code
    if status_code.startswith("5"):
        level = "ERROR"
    elif status_code.startswith("4"):
        level = "WARN"
    else:
        level = "INFO"
        
    message = f"{data.get('method', '')} {data.get('url', '')} - {status_code}"
    
    return {
        "timestamp": data["timestamp"],
        "timestamp_iso": parsed_time.isoformat(),
        "timestamp_unix": parsed_time.timestamp(),
        "month": parsed_time.strftime("%B"),
        "year": parsed_time.year,
        "day": parsed_time.day,
        "level": level,
        "component": "Apache",
        "has_exception": False,
        "message_preview": message[:50],
        "message": message,
        "ip": data.get("ip", ""),
        "status": status_code,
        "size": data.get("size", ""),
        "http_method": data.get("method", ""),
        "url": data.get("url", ""),
        "raw": entry.strip()
    }

def process_json_log(entry: str, format_info: Dict) -> Optional[Dict]:
    """Process a JSON format log entry"""
    try:
        import json
        data = json.loads(entry.strip())
        
        # Extract common fields from JSON
        timestamp = data.get("timestamp", data.get("time", data.get("@timestamp", "")))
        level = data.get("level", data.get("severity", data.get("log_level", "UNKNOWN"))).upper()
        message = data.get("message", data.get("msg", ""))
        component = data.get("component", data.get("service", data.get("logger", "Unknown")))
        
        # Try to parse timestamp if present
        parsed_time = None
        if timestamp:
            # Try common timestamp formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds
                "%Y-%m-%dT%H:%M:%SZ",     # ISO format without microseconds
                "%Y-%m-%d %H:%M:%S.%f",   # Standard format with microseconds
                "%Y-%m-%d %H:%M:%S"       # Standard format without microseconds
            ]
            
            for fmt in formats:
                try:
                    parsed_time = datetime.strptime(timestamp, fmt)
                    break
                except Exception:
                    continue
        
        if not parsed_time:
            # If we couldn't parse the timestamp, use current time
            parsed_time = datetime.now()
            
        # Build metadata dictionary
        metadata = {
            "timestamp": timestamp if timestamp else parsed_time.isoformat(),
            "timestamp_iso": parsed_time.isoformat(),
            "timestamp_unix": parsed_time.timestamp(),
            "month": parsed_time.strftime("%B"),
            "year": parsed_time.year,
            "day": parsed_time.day,
            "level": level,
            "component": component,
            "has_exception": "error" in data or "exception" in data or "stack" in data or "stacktrace" in data,
            "message_preview": message[:50] if message else "",
            "message": message,
            "raw": entry.strip()
        }
        
        # Add all original JSON fields to metadata
        for key, value in data.items():
            if key not in metadata and isinstance(value, (str, int, float, bool)) and value is not None:
                metadata[key] = value
                
        return metadata
        
    except Exception as e:
        print(f"Error processing JSON log: {str(e)}")
        return None

def process_syslog(entry: str, format_info: Dict) -> Optional[Dict]:
    """Process a syslog format log entry"""
    match = format_info["regex"].match(entry.strip())
    if not match:
        return None
        
    data = match.groupdict()
    
    # Syslog timestamp doesn't include the year, so we'll use current year
    timestamp = data["timestamp"]
    current_year = datetime.now().year
    try:
        parsed_time = datetime.strptime(f"{timestamp} {current_year}", "%b %d %H:%M:%S %Y")
        # Handle case where the log is from previous year
        if parsed_time > datetime.now():
            parsed_time = datetime.strptime(f"{timestamp} {current_year-1}", "%b %d %H:%M:%S %Y")
    except Exception:
        return None
    
    message = data.get("message", "").strip()
    process = data.get("process", "Unknown")
    
    # Try to determine log level from message
    if "error" in message.lower() or "fatal" in message.lower() or "critical" in message.lower():
        level = "ERROR"
    elif "warn" in message.lower():
        level = "WARN"
    else:
        level = "INFO"
        
    return {
        "timestamp": timestamp,
        "timestamp_iso": parsed_time.isoformat(),
        "timestamp_unix": parsed_time.timestamp(),
        "month": parsed_time.strftime("%B"),
        "year": parsed_time.year,
        "day": parsed_time.day,
        "level": level,
        "component": process,
        "process": process,
        "pid": data.get("pid", ""),
        "hostname": data.get("hostname", ""),
        "has_exception": False,
        "message_preview": message[:50],
        "message": message,
        "raw": entry.strip()
    }

# Map format IDs to their processing functions
FORMAT_PROCESSORS = {
    "standard": process_standard_log,
    "apache": process_apache_log,
    "json": process_json_log,
    "syslog": process_syslog
}

def parse_log_entry(entry: str, format_id: str = "standard"):
    """Parse a single log entry using the specified format"""
    if format_id not in LOG_FORMATS:
        format_id = "standard"
        
    format_info = LOG_FORMATS[format_id]
    processor = FORMAT_PROCESSORS.get(format_id)
    
    if not processor:
        return None
        
    metadata = processor(entry, format_info)
    if metadata:
        return Document(
            page_content=metadata.get("message", entry.strip()),
            metadata=metadata
        )
    return None

def process_logs(file_path: str, format_id: str = None):
    """Process logs from file with format auto-detection if format_id not provided"""
    documents = []
    try:
        with open(file_path, "r") as file:
            content = file.read()
            
        # Auto-detect format if not provided
        if not format_id or format_id not in LOG_FORMATS:
            format_id = detect_log_format(content)
            print(f"Auto-detected log format: {LOG_FORMATS[format_id]['name']}")
            
        format_info = LOG_FORMATS[format_id]
        
        # For JSON logs, handle differently
        if format_id == "json":
            lines = content.strip().split("\n")
            for line in lines:
                if line.strip():
                    doc = parse_log_entry(line, format_id)
                    if doc:
                        documents.append(doc)
        else:
            # For regex-based formats
            entry_pattern = format_info["entry_split_pattern"]
            entries = re.split(entry_pattern, content)
            
            for entry in entries:
                if entry.strip():
                    doc = parse_log_entry(entry, format_id)
                    if doc:
                        documents.append(doc)
                        
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        
    return documents

def get_available_formats():
    """Return information about available log formats"""
    return {
        format_id: {
            "name": info["name"],
            "description": info["description"]
        }
        for format_id, info in LOG_FORMATS.items()
    }
