"""
Document loading and processing for AtlasAI.

Handles PDF and DOCX documents with text cleaning and chunking.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading, cleaning, and chunking."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    
    @staticmethod
    def strip_boilerplate(text: str) -> str:
        """
        Remove common footer/header boilerplate from PDFs.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text with boilerplate removed
        """
        if not text:
            return text
        
        patterns = [
            r"\bProprietary\s*-\s*See\s*Copyright\s*Page\b",
            r"\bContents\b",
            r"\bADMS\s*[\d\.]+\s*Modeling\s*Overview\s*and\s*Converter\s*User\s*Guide\b",
            r"\bDistribution\s*Model\s*Manager\s*User\s*Guide\b",
            r"^\s*Page\s*\d+\s*$",
        ]
        
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        
        # Remove extra spaces
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned
    
    def load_documents(
        self, 
        pdf_paths: List[Path], 
        docx_paths: Optional[List[Path]] = None
    ) -> tuple[List[Document], List[str]]:
        """
        Load documents from file paths.
        
        Args:
            pdf_paths: List of PDF file paths
            docx_paths: List of DOCX file paths (optional)
            
        Returns:
            Tuple of (loaded documents, list of missing files)
        """
        docs = []
        missing = []
        docx_paths = docx_paths or []
        
        # Load PDFs
        for p in pdf_paths:
            if not p.exists():
                missing.append(str(p))
            else:
                try:
                    docs.extend(PyPDFLoader(str(p)).load())
                except Exception as e:
                    st.warning(f"Failed to read PDF {p.name}: {e}")
        
        # Load DOCX files
        for p in docx_paths:
            if not p.exists():
                missing.append(str(p))
            else:
                try:
                    docs.extend(Docx2txtLoader(str(p)).load())
                except Exception as e:
                    st.warning(f"Failed to read DOCX {p.name}: {e}")
        
        return docs, missing
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        return self.splitter.split_documents(documents)
    
    def compute_document_hash(self, file_paths: List[Path]) -> str:
        """
        Compute hash of document files for cache invalidation.
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            MD5 hash of all file contents
        """
        hasher = hashlib.md5()
        
        for path in sorted(file_paths):  # Sort for consistency
            if path.exists():
                # Hash file path and modification time
                hasher.update(str(path).encode())
                hasher.update(str(path.stat().st_mtime).encode())
        
        # Also hash chunk settings to invalidate cache if they change
        hasher.update(f"{self.chunk_size}:{self.chunk_overlap}".encode())
        
        return hasher.hexdigest()
