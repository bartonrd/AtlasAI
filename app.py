"""
AtlasAI - Modular RAG Chatbot Application

Improved architecture with:
- Modular design with separation of concerns
- Persistent vector store caching
- Better model support and configuration
- Enhanced error handling
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid
import streamlit as st

# Import modular components
from atlasai_core.config import config
from atlasai_core.document_processor import DocumentProcessor
from atlasai_core.vector_store import VectorStoreManager
from atlasai_core.llm_manager import LLMManager
from atlasai_core.rag_chain import RAGChain
from atlasai_core.utils import thinking_message, to_bullets, generate_chat_name


# ---------------------------
# Session State Initialization
# ---------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    
    if "current_chat_id" not in st.session_state:
        initial_id = str(uuid.uuid4())
        st.session_state.chats[initial_id] = {
            "name": config.default_chat_name,
            "created_at": datetime.now(),
            "messages": []
        }
        st.session_state.current_chat_id = initial_id
    
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 1
    
    # Initialize settings from config
    if "top_k" not in st.session_state:
        st.session_state.top_k = config.top_k
    
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = config.chunk_size
    
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = config.chunk_overlap
    
    # Initialize managers (lazy loading)
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = None
    
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = None


# ---------------------------
# Helper Functions for Chat Management
# ---------------------------
def create_new_chat():
    """Create a new chat instance"""
    st.session_state.chat_counter += 1
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {
        "name": config.default_chat_name,
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


def validate_settings(top_k, chunk_size, chunk_overlap):
    """
    Validate chatbot settings and return error messages if any.
    
    Args:
        top_k: Number of chunks to retrieve
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    if top_k < 1:
        errors.append("Top K must be a positive integer (minimum 1)")
    elif top_k > 20:
        errors.append("Top K should not exceed 20 for optimal performance")
    
    if chunk_size < 100:
        errors.append("Chunk Size must be at least 100 characters")
    elif chunk_size > 2000:
        errors.append("Chunk Size should not exceed 2000 to fit within model token limits")
    
    if chunk_overlap < 0:
        errors.append("Chunk Overlap must be a non-negative integer")
    elif chunk_overlap >= chunk_size:
        errors.append("Chunk Overlap must be less than Chunk Size")
    elif chunk_overlap > chunk_size * config.max_overlap_percentage:
        max_percent = int(config.max_overlap_percentage * 100)
        errors.append(f"Chunk Overlap should not exceed {max_percent}% of Chunk Size for best results")
    
    return errors


# ---------------------------
# Document and RAG Setup
# ---------------------------
def get_document_paths() -> tuple[list[Path], list[Path]]:
    """
    Get all document paths (default + uploaded).
    
    Returns:
        Tuple of (pdf_paths, docx_paths)
    """
    pdf_paths = list(config.default_pdf_paths)
    docx_paths = []
    
    # Add uploaded documents
    temp_dir = Path(os.getcwd()) / "tmp_docs"
    for doc_name in st.session_state.uploaded_documents:
        doc_path = temp_dir / doc_name
        if doc_path.exists():
            if doc_path.suffix.lower() == ".pdf":
                pdf_paths.append(doc_path)
            elif doc_path.suffix.lower() == ".docx":
                docx_paths.append(doc_path)
    
    return pdf_paths, docx_paths


def process_query(prompt_text: str, thinking_placeholder):
    """
    Process a user query through the RAG pipeline.
    
    Args:
        prompt_text: User's question
        thinking_placeholder: Streamlit placeholder for status messages
        
    Returns:
        Tuple of (formatted_answer, source_list)
    """
    # Get document paths
    pdf_paths, docx_paths = get_document_paths()
    
    # Initialize document processor
    thinking_placeholder.markdown(thinking_message("Initializing..."), unsafe_allow_html=True)
    doc_processor = DocumentProcessor(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap
    )
    
    # Load documents
    thinking_placeholder.markdown(thinking_message("Loading documents..."), unsafe_allow_html=True)
    docs, missing = doc_processor.load_documents(pdf_paths, docx_paths)
    
    if missing:
        thinking_placeholder.empty()
        st.error("The following files were not found:\n- " + "\n- ".join(missing))
        st.stop()
    
    if not docs:
        thinking_placeholder.empty()
        st.error("No documents loaded. Please add PDF/DOCX files.")
        st.stop()
    
    # Split documents
    thinking_placeholder.markdown(thinking_message("Processing documents..."), unsafe_allow_html=True)
    splits = doc_processor.split_documents(docs)
    
    if not splits:
        thinking_placeholder.empty()
        st.error("No text chunks produced. If files are scanned images, run OCR first.")
        st.stop()
    
    # Compute document hash for caching
    doc_hash = doc_processor.compute_document_hash(pdf_paths + docx_paths)
    
    # Initialize vector store manager
    thinking_placeholder.markdown(thinking_message("Creating embeddings..."), unsafe_allow_html=True)
    vector_store_manager = VectorStoreManager(
        embedding_model_path=config.embedding_model,
        cache_dir=config.vector_store_path,
        use_cache=config.use_persistent_cache
    )
    
    try:
        vector_store_manager.initialize_embeddings()
    except RuntimeError as e:
        # Try fallback model
        st.info(f"Trying fallback embedding model: {config.fallback_embedding_model}")
        vector_store_manager = VectorStoreManager(
            embedding_model_path=config.fallback_embedding_model,
            cache_dir=config.vector_store_path,
            use_cache=config.use_persistent_cache
        )
        vector_store_manager.initialize_embeddings()
    
    # Create vector store (with caching)
    vectorstore = vector_store_manager.create_vector_store(splits, doc_hash)
    retriever = vector_store_manager.get_retriever(k=st.session_state.top_k)
    
    # Initialize LLM
    thinking_placeholder.markdown(thinking_message("Loading language model..."), unsafe_allow_html=True)
    llm_manager = LLMManager(
        model_path=config.text_gen_model,
        fallback_model=config.fallback_text_gen_model,
        max_new_tokens=config.max_new_tokens,
        use_sampling=config.use_sampling,
        temperature=config.temperature
    )
    
    llm = llm_manager.initialize_llm()
    
    # Build RAG chain
    thinking_placeholder.markdown(thinking_message("Setting up RAG chain..."), unsafe_allow_html=True)
    rag_chain = RAGChain(llm=llm, retriever=retriever)
    rag_chain.build_chain()
    
    # Run query
    thinking_placeholder.markdown(thinking_message("Generating answer..."), unsafe_allow_html=True)
    result = rag_chain.query(prompt_text)
    
    # Format results
    answer = result.get("result", "")
    sources = result.get("source_documents", [])
    
    formatted_answer = to_bullets(answer)
    source_list = RAGChain.format_sources(sources) if sources else []
    
    return formatted_answer, source_list


# ---------------------------
# Initialize
# ---------------------------
initialize_session_state()

# ---------------------------
# Streamlit UI - Sidebar
# ---------------------------
st.set_page_config(page_title="AtlasAI Chat", layout="wide")

with st.sidebar:
    st.title("AtlasAI")
    st.caption("v2.0 - Modular Architecture")
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["💬 Chats", "⚙️ Settings", "📄 Documents"])
    
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
        st.subheader("Settings")
        st.write("Adjust the chatbot configuration below:")
        
        # Top K setting
        top_k_input = st.number_input(
            "Top K (number of chunks retrieved)",
            min_value=1,
            max_value=20,
            value=st.session_state.top_k,
            step=1,
            help="Number of text chunks to retrieve from documents for each query."
        )
        
        # Chunk Size setting
        chunk_size_input = st.number_input(
            "Chunk Size (characters)",
            min_value=100,
            max_value=2000,
            value=st.session_state.chunk_size,
            step=50,
            help="Size of each text chunk in characters."
        )
        
        # Chunk Overlap setting
        max_overlap = min(chunk_size_input - 1, config.max_chunk_overlap)
        current_overlap = min(st.session_state.chunk_overlap, max_overlap)
        
        chunk_overlap_input = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=max_overlap,
            value=current_overlap,
            step=10,
            help="Overlap between consecutive chunks in characters."
        )
        
        # Validate settings
        validation_errors = validate_settings(top_k_input, chunk_size_input, chunk_overlap_input)
        has_validation_errors = len(validation_errors) > 0
        
        # Apply button
        apply_button = st.button("Apply Settings", use_container_width=True, disabled=has_validation_errors)
        
        if apply_button and not has_validation_errors:
            st.session_state.top_k = top_k_input
            st.session_state.chunk_size = chunk_size_input
            st.session_state.chunk_overlap = chunk_overlap_input
            st.success("Settings applied successfully!")
            st.rerun()
        
        # Display validation errors
        if has_validation_errors:
            st.divider()
            error_message = "**Settings Validation Errors:**\n\n" + "\n".join([f"❌ {error}" for error in validation_errors])
            st.error(error_message)
        
        st.divider()
        st.write("**Current Active Settings:**")
        st.text(f"Top K: {st.session_state.top_k}")
        st.text(f"Chunk Size: {st.session_state.chunk_size}")
        st.text(f"Chunk Overlap: {st.session_state.chunk_overlap}")
        st.text(f"Cache Enabled: {config.use_persistent_cache}")
    
    with tab3:
        st.subheader("Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        temp_dir = Path(os.getcwd()) / "tmp_docs"
        temp_dir.mkdir(exist_ok=True)
        
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name not in st.session_state.uploaded_documents:
                    save_path = temp_dir / uf.name
                    with open(save_path, "wb") as f:
                        f.write(uf.read())
                    st.session_state.uploaded_documents.append(uf.name)
        
        st.divider()
        st.write("**Loaded Documents:**")
        
        # Show default documents
        st.write("*Default documents:*")
        for doc_path in config.default_pdf_paths:
            st.text(f"📄 {doc_path.name}")
        
        # Show uploaded documents
        if st.session_state.uploaded_documents:
            st.write("*Uploaded documents:*")
            for doc_name in st.session_state.uploaded_documents:
                st.text(f"📎 {doc_name}")

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
                    st.write(source_info)

# Chat input
prompt_text = st.chat_input("Ask a question about your documents...")

if prompt_text:
    # Add user message
    current_chat["messages"].append({
        "role": "user",
        "content": prompt_text
    })
    
    # Auto-name chat based on first message
    if len(current_chat["messages"]) == 1 and current_chat["name"] == config.default_chat_name:
        current_chat["name"] = generate_chat_name(prompt_text)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt_text)
    
    # Process query and display response
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(thinking_message("Thinking..."), unsafe_allow_html=True)
        
        try:
            formatted_answer, source_list = process_query(prompt_text, thinking_placeholder)
            
            # Display answer
            thinking_placeholder.markdown(formatted_answer)
            
            # Display sources
            if source_list:
                with st.expander("Sources"):
                    for source_info in source_list:
                        st.write(source_info)
            
            # Add to chat history
            current_chat["messages"].append({
                "role": "assistant",
                "content": formatted_answer,
                "sources": source_list if source_list else []
            })
            
        except Exception as e:
            thinking_placeholder.empty()
            st.error(f"An error occurred: {str(e)}")
            # Log full traceback for debugging (not displayed to user)
            import traceback
            import logging
            logging.error("RAG query failed", exc_info=True)
