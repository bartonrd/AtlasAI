"""
Chat interface module - handles Streamlit UI
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st

from ..config import Settings
from ..services import RAGService
from ..utils import format_answer_as_bullets, thinking_message


class ChatInterface:
    """Manages the chat UI and state"""
    
    def __init__(self, settings: Settings, rag_service: RAGService):
        """
        Initialize chat interface
        
        Args:
            settings: Application settings
            rag_service: RAG service instance
        """
        self.settings = settings
        self.rag_service = rag_service
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "chats" not in st.session_state:
            st.session_state.chats = {}
        
        if "current_chat_id" not in st.session_state:
            initial_id = str(uuid.uuid4())
            st.session_state.chats[initial_id] = {
                "name": self.settings.ui.default_chat_name,
                "created_at": datetime.now(),
                "messages": []
            }
            st.session_state.current_chat_id = initial_id
        
        if "uploaded_documents" not in st.session_state:
            st.session_state.uploaded_documents = []
        
        if "chat_counter" not in st.session_state:
            st.session_state.chat_counter = 1
        
        # Initialize settings in session state
        if "top_k" not in st.session_state:
            st.session_state.top_k = self.settings.rag.top_k
        
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = self.settings.rag.chunk_size
        
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = self.settings.rag.chunk_overlap
    
    def render(self):
        """Render the complete chat interface"""
        st.set_page_config(
            page_title=self.settings.ui.page_title,
            layout=self.settings.ui.layout
        )
        
        with st.sidebar:
            self._render_sidebar()
        
        self._render_main_chat()
    
    def _render_sidebar(self):
        """Render sidebar with tabs"""
        st.title("AtlasAI")
        
        tab1, tab2, tab3 = st.tabs(["💬 Chats", "⚙️ Settings", "📄 Documents"])
        
        with tab1:
            self._render_chats_tab()
        
        with tab2:
            self._render_settings_tab()
        
        with tab3:
            self._render_documents_tab()
    
    def _render_chats_tab(self):
        """Render chats management tab"""
        st.subheader("Chat Sessions")
        
        if st.button("➕ New Chat", use_container_width=True):
            self._create_new_chat()
            st.rerun()
        
        st.divider()
        
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
                        self._delete_chat(chat_id)
                        st.rerun()
            
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
                            self._rename_chat(chat_id, new_name.strip())
                        st.session_state[f"renaming_{chat_id}"] = False
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                        st.session_state[f"renaming_{chat_id}"] = False
                        st.rerun()
    
    def _render_settings_tab(self):
        """Render settings tab"""
        st.subheader("Settings")
        st.write("Adjust the chatbot configuration below:")
        
        top_k_input = st.number_input(
            "Top K (number of chunks retrieved)",
            min_value=1,
            max_value=self.settings.rag.max_top_k,
            value=st.session_state.top_k,
            step=1,
            help="Number of text chunks to retrieve from documents for each query"
        )
        
        chunk_size_input = st.number_input(
            "Chunk Size (characters)",
            min_value=self.settings.rag.min_chunk_size,
            max_value=self.settings.rag.max_chunk_size,
            value=st.session_state.chunk_size,
            step=50,
            help="Size of each text chunk in characters"
        )
        
        max_overlap = min(chunk_size_input - 1, int(chunk_size_input * self.settings.rag.max_overlap_percentage))
        current_overlap = min(st.session_state.chunk_overlap, max_overlap)
        
        chunk_overlap_input = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=max_overlap,
            value=current_overlap,
            step=10,
            help="Overlap between consecutive chunks in characters"
        )
        
        validation_errors = self.settings.rag.validate()
        has_validation_errors = len(validation_errors) > 0
        
        apply_button = st.button(
            "Apply Settings", 
            use_container_width=True, 
            disabled=has_validation_errors
        )
        
        if apply_button and not has_validation_errors:
            st.session_state.top_k = top_k_input
            st.session_state.chunk_size = chunk_size_input
            st.session_state.chunk_overlap = chunk_overlap_input
            self.rag_service.update_rag_settings(top_k_input, chunk_size_input, chunk_overlap_input)
            st.success("Settings applied successfully!")
            st.rerun()
        
        if has_validation_errors:
            st.divider()
            error_message = "**Settings Validation Errors:**\n\n" + "\n".join(
                [f"❌ {error}" for error in validation_errors]
            )
            st.error(error_message)
        
        st.divider()
        st.write("**Current Active Settings:**")
        st.text(f"Top K: {st.session_state.top_k}")
        st.text(f"Chunk Size: {st.session_state.chunk_size}")
        st.text(f"Chunk Overlap: {st.session_state.chunk_overlap}")
    
    def _render_documents_tab(self):
        """Render documents tab"""
        st.subheader("Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        temp_dir = self.settings.paths.temp_dir
        
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name not in st.session_state.uploaded_documents:
                    save_path = temp_dir / uf.name
                    with open(save_path, "wb") as f:
                        f.write(uf.read())
                    st.session_state.uploaded_documents.append(uf.name)
        
        st.divider()
        st.write("**Loaded Documents:**")
        
        st.write("*Default documents:*")
        for doc_path in self.settings.paths.get_default_documents():
            st.text(f"📄 {doc_path.name}")
        
        if st.session_state.uploaded_documents:
            st.write("*Uploaded documents:*")
            for doc_name in st.session_state.uploaded_documents:
                st.text(f"📎 {doc_name}")
    
    def _render_main_chat(self):
        """Render main chat interface"""
        current_chat = self._get_current_chat()
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
            self._handle_chat_input(prompt_text, current_chat)
    
    def _handle_chat_input(self, prompt_text: str, current_chat: Dict[str, Any]):
        """Handle user input and generate response"""
        # Add user message
        current_chat["messages"].append({
            "role": "user",
            "content": prompt_text
        })
        
        # Auto-name chat
        if len(current_chat["messages"]) == 1 and current_chat["name"] == self.settings.ui.default_chat_name:
            current_chat["name"] = self._generate_chat_name(prompt_text)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt_text)
        
        # Generate response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            
            try:
                # Get document paths
                document_paths = self._get_document_paths()
                
                if not document_paths:
                    st.error("No documents available. Please add documents in the Documents tab.")
                    return
                
                # Show processing status
                thinking_placeholder.markdown(thinking_message("Processing..."), unsafe_allow_html=True)
                
                # Query RAG service
                result = self.rag_service.query(prompt_text, document_paths)
                
                # Format answer
                formatted_answer = format_answer_as_bullets(result["answer"])
                
                # Display answer
                thinking_placeholder.markdown(formatted_answer)
                
                # Display sources
                if result["sources"]:
                    with st.expander("Sources"):
                        for source_info in result["sources"]:
                            st.write(source_info)
                
                # Add to history
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": formatted_answer,
                    "sources": result["sources"]
                })
                
            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"Error generating response: {e}")
    
    def _get_document_paths(self) -> List[Path]:
        """Get list of all document paths"""
        paths = self.settings.paths.get_default_documents()
        
        # Add uploaded documents
        temp_dir = self.settings.paths.temp_dir
        for doc_name in st.session_state.uploaded_documents:
            doc_path = temp_dir / doc_name
            if doc_path.exists():
                paths.append(doc_path)
        
        return paths
    
    def _create_new_chat(self):
        """Create a new chat"""
        st.session_state.chat_counter += 1
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "name": self.settings.ui.default_chat_name,
            "created_at": datetime.now(),
            "messages": []
        }
        st.session_state.current_chat_id = new_id
    
    def _delete_chat(self, chat_id: str):
        """Delete a chat"""
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[chat_id]
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
    
    def _rename_chat(self, chat_id: str, new_name: str):
        """Rename a chat"""
        if chat_id in st.session_state.chats:
            st.session_state.chats[chat_id]["name"] = new_name
    
    def _get_current_chat(self) -> Dict[str, Any]:
        """Get current chat"""
        return st.session_state.chats[st.session_state.current_chat_id]
    
    def _generate_chat_name(self, message: str, max_length: int = 30) -> str:
        """Generate chat name from first message"""
        if not message:
            return self.settings.ui.default_chat_name
        
        clean_msg = " ".join(message.split())
        
        if len(clean_msg) <= max_length:
            return clean_msg
        
        ellipsis = "..."
        max_text_length = max_length - len(ellipsis)
        truncated = clean_msg[:max_text_length]
        space_pos = truncated.rfind(' ')
        
        if space_pos > 0:
            return truncated[:space_pos] + ellipsis
        else:
            return truncated + ellipsis
