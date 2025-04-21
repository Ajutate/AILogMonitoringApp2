# === log_processor.py ===
import re
from langchain.schema import Document
from datetime import datetime
# Finalized regex pattern
log_regex = re.compile(r"""
    ^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})          # Timestamp
    \s+
    (?:\[(?P<level1>[A-Z]+)\]|(?P<level2>[A-Z]+))                 # [LEVEL] or LEVEL
    (?:\s+\[(?P<component>[^\]]+)\])?                             # Optional [Component]
    \s+(?P<message>[^\n]+)                                        # First line of message
    (?P<exception>(?:\n\s+(?!\d{4}-\d{2}-\d{2}).*)*)               # Multi-line stack trace
    """, re.VERBOSE | re.MULTILINE)

#This method not used in the code but can be used to parse log entries
def parse_log_entry(entry: str):
    match = log_regex.match(entry.strip())
    print(f"Processing entry: {entry.strip()}");
    print(f"match: {match}");
    if not match:
        return None

    timestamp = match.group("timestamp")
    level = match.group("level1") or match.group("level2")
    component = match.group("component") or "Unknown"
    message = match.group("message").strip()
    exception = match.group("exception").strip() if match.group("exception") else ""
    print(f"timestamp: {timestamp}");
    print(f"level: {level}");
    full_content = f"{message}\n{exception}" if exception else message

    return Document(
        page_content=full_content,
        metadata={
            "timestamp": timestamp,
            "level": level,
            "component": component,
            "message": message,
            "exception": exception,
        }
    )

#not in use
def split_log_entries(raw_log_data: str):
    return re.split(r"(?=^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", raw_log_data, flags=re.MULTILINE)

#not in use
def process_logsold(file_path: str):
    with open(file_path, "r") as file:
        raw_data = file.read()

    entries = split_log_entries(raw_data)
    documents = [parse_log_entry(entry) for entry in entries]
    return [doc for doc in documents if doc is not None]

def process_logs(file_path: str):
    documents = []
    with open(file_path, "r") as file:
        content = file.read()

    entries = re.split(r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content)

    for entry in entries:
        match = log_regex.match(entry.strip())
        if match:
            data = match.groupdict()
            try:
                parsed_time = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue  # skip if timestamp parsing fails

            print(f"parsed_time: {parsed_time}");
            level = match.group("level1") or match.group("level2") or "UNKNOWN"  # Ensure level is never None
            print(f"level: {level}");
            message = match.group("message").strip()
            exception = match.group("exception").strip() if match.group("exception") else ""
            full_content = f"{message}\n{exception}" if exception else message
            timestamp = match.group("timestamp")
            # Ensure all metadata values are valid types (str, int, float, bool) and not None
            metadata = {
                "timestamp": timestamp,
                "timestamp_iso": parsed_time.isoformat(),
                "timestamp_unix": parsed_time.timestamp(),  # Add Unix timestamp as numeric value
                "month": parsed_time.strftime("%B"),  # Full month name like "January"
                "year": parsed_time.year,
                "day": parsed_time.day,
                "level": level,
                "component": data.get("component", "") or "Unknown",  # Ensure component is never None
                "has_exception": bool(exception),  # Add boolean flag instead of actual exception text
                "message_preview": message[:50] if message else "",  # Add a preview of the message
                "raw": entry.strip()  # Add the raw log entry
            }
            documents.append(Document(page_content=entry.strip(), metadata=metadata))

    return documents
