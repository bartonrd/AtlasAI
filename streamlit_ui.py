"""
AtlasAI Streamlit UI - connects to the FastAPI runtime backend
"""

import os
import streamlit as st
import requests
from datetime import datetime
import uuid

# Configuration
RUNTIME_HOST = os.getenv("ATLASAI_RUNTIME_HOST", "127.0.0.1")
RUNTIME_PORT = int(os.getenv("ATLASAI_RUNTIME_PORT", "8000"))
RUNTIME_BASE_URL = f"http://{RUNTIME_HOST}:{RUNTIME_PORT}"

# Timeout constants
HEALTH_CHECK_TIMEOUT = 5
CHAT_REQUEST_TIMEOUT = 120

DEFAULT_CHAT_NAME = "New Chat"

# HTTP session for connection reuse
http_session = requests.Session()

# ---------------------------
# Session State Initialization
# ---------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    if "chats" not in st.session_state:
        # Dictionary to store all chat instances
        st.session_state.chats = {}
    
    if "current_chat_id" not in st.session_state:
        # Create initial chat
        initial_id = str(uuid.uuid4())
        st.session_state.chats[initial_id] = {
            "name": DEFAULT_CHAT_NAME,
            "created_at": datetime.now(),
            "messages": []
        }
        st.session_state.current_chat_id = initial_id
    
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 1

# Initialize session state
initialize_session_state()

# ---------------------------
# Helper Functions for Chat Management
# ---------------------------
def generate_chat_name(message: str, max_length: int = 30) -> str:
    """Generate a short chat name from a message"""
    if not message:
        return DEFAULT_CHAT_NAME
    
    # Remove extra whitespace and newlines
    clean_msg = " ".join(message.split())
    
    # Truncate to max_length
    if len(clean_msg) <= max_length:
        return clean_msg
    
    # Truncate and add ellipsis
    ellipsis = "..."
    max_text_length = max_length - len(ellipsis)
    truncated = clean_msg[:max_text_length]
    # Try to break at word boundary
    space_pos = truncated.rfind(' ')
    if space_pos > 0:
        return truncated[:space_pos] + ellipsis
    else:
        return truncated + ellipsis

def create_new_chat():
    """Create a new chat instance"""
    st.session_state.chat_counter += 1
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {
        "name": DEFAULT_CHAT_NAME,
        "created_at": datetime.now(),
        "messages": []
    }
    st.session_state.current_chat_id = new_id

def delete_chat(chat_id):
    """Delete a chat instance"""
    if len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

def rename_chat(chat_id, new_name):
    """Rename a chat instance"""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["name"] = new_name

def get_current_chat():
    """Get the current chat instance"""
    return st.session_state.chats[st.session_state.current_chat_id]

# ---------------------------
# API Communication
# ---------------------------
def check_runtime_health():
    """Check if the runtime is healthy"""
    try:
        response = http_session.get(f"{RUNTIME_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy", data.get("message", "")
        return False, "Runtime not responding"
    except requests.RequestException:
        return False, "Unable to connect to runtime. Please ensure it's running on localhost:8000"

def send_chat_message(message: str):
    """Send a chat message to the runtime and get response"""
    try:
        response = http_session.post(
            f"{RUNTIME_BASE_URL}/chat",
            json={"message": message},
            timeout=CHAT_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error("Unable to get response from runtime. Please check the runtime status.")
        return None

# ---------------------------
# Streamlit UI - Sidebar
# ---------------------------
st.set_page_config(page_title="AtlasAI Chat", layout="wide")

with st.sidebar:
    st.title("AtlasAI")
    
    # Check runtime health
    is_healthy, health_message = check_runtime_health()
    if is_healthy:
        st.success(f"✓ Runtime Ready")
    else:
        st.error(f"✗ Runtime Offline: {health_message}")
        st.info("Make sure the Python runtime is running on localhost:8000")
    
    st.divider()
    
    # Tab navigation
    tab1, tab2 = st.tabs(["💬 Chats", "📄 Info"])
    
    with tab1:
        st.subheader("Chat Sessions")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()
        
        st.divider()
        
        # List all chats
        for chat_id, chat_data in st.session_state.chats.items():
            is_current = chat_id == st.session_state.current_chat_id
            
            # Create columns for chat button and delete button
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                if st.button(
                    chat_data["name"],
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            
            with col2:
                # Edit button
                if st.button("✏️", key=f"edit_{chat_id}"):
                    st.session_state[f"renaming_{chat_id}"] = True
                    st.rerun()
            
            with col3:
                if len(st.session_state.chats) > 1:
                    if st.button("🗑️", key=f"del_{chat_id}"):
                        delete_chat(chat_id)
                        st.rerun()
            
            # Show rename input if edit was clicked
            if st.session_state.get(f"renaming_{chat_id}", False):
                new_name = st.text_input(
                    "New name:",
                    value=chat_data["name"],
                    key=f"rename_input_{chat_id}",
                    max_chars=50
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save", key=f"save_{chat_id}", use_container_width=True):
                        if new_name.strip():
                            rename_chat(chat_id, new_name.strip())
                        st.session_state[f"renaming_{chat_id}"] = False
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                        st.session_state[f"renaming_{chat_id}"] = False
                        st.rerun()
    
    with tab2:
        st.subheader("About")
        st.write("AtlasAI - RAG Chatbot")
        st.write(f"Runtime: {RUNTIME_BASE_URL}")
        st.write("Mode: Streamlit UI")
        st.divider()
        st.write("**Features:**")
        st.write("- Multiple chat sessions")
        st.write("- Document-based Q&A")
        st.write("- Source citations")

# ---------------------------
# Main Chat UI
# ---------------------------
current_chat = get_current_chat()
st.title(f"💬 {current_chat['name']}")

# Display chat history
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for source_info in message["sources"]:
                    st.write(f"{source_info['index']}. {source_info['source']} (page {source_info['page']})")

# Chat input
prompt_text = st.chat_input("Ask a question about your documents...", disabled=not is_healthy)

if prompt_text:
    # Add user message to chat history
    current_chat["messages"].append({
        "role": "user",
        "content": prompt_text
    })
    
    # Auto-name chat based on first message if still using default name
    if len(current_chat["messages"]) == 1 and current_chat["name"] == DEFAULT_CHAT_NAME:
        current_chat["name"] = generate_chat_name(prompt_text)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response_data = send_chat_message(prompt_text)
            
            if response_data:
                answer = response_data.get("answer", "")
                sources = response_data.get("sources", [])
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"{source['index']}. {source['source']} (page {source['page']})")
                
                # Add assistant message to chat history
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                error_msg = "Failed to get response from runtime. Please check the runtime status."
                st.error(error_msg)
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
