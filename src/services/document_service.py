"""
Document processing service - handles loading and splitting documents
"""

import os
import re
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentService:
    """Service for loading and processing documents"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize document service
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    
    @staticmethod
    def strip_boilerplate(text: str) -> str:
        """
        Remove common footer/header boilerplate from documents
        
        Args:
            text: Raw text to clean
            
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
    
    def load_pdf(self, path: Path) -> List[Document]:
        """
        Load a PDF document
        
        Args:
            path: Path to PDF file
            
        Returns:
            List of loaded documents
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF cannot be loaded
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        
        try:
            loader = PyPDFLoader(str(path))
            return loader.load()
        except Exception as e:
            raise ValueError(f"Failed to load PDF {path}: {e}")
    
    def load_docx(self, path: Path) -> List[Document]:
        """
        Load a DOCX document
        
        Args:
            path: Path to DOCX file
            
        Returns:
            List of loaded documents
            
        Raises:
            FileNotFoundError: If DOCX doesn't exist
            ValueError: If DOCX cannot be loaded
        """
        if not path.exists():
            raise FileNotFoundError(f"DOCX not found: {path}")
        
        try:
            loader = Docx2txtLoader(str(path))
            return loader.load()
        except Exception as e:
            raise ValueError(f"Failed to load DOCX {path}: {e}")
    
    def load_document(self, path: Path) -> List[Document]:
        """
        Load a document (auto-detect type from extension)
        
        Args:
            path: Path to document
            
        Returns:
            List of loaded documents
        """
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(path)
        elif ext == ".docx":
            return self.load_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def load_documents(self, paths: List[Path]) -> tuple[List[Document], List[str]]:
        """
        Load multiple documents
        
        Args:
            paths: List of document paths
            
        Returns:
            Tuple of (loaded documents, list of errors)
        """
        documents = []
        errors = []
        
        for path in paths:
            try:
                docs = self.load_document(path)
                documents.extend(docs)
            except Exception as e:
                errors.append(f"{path.name}: {str(e)}")
        
        return documents, errors
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split document chunks
            
        Raises:
            ValueError: If no chunks are produced
        """
        if not documents:
            raise ValueError("No documents provided for splitting")
        
        splits = self._splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("No text chunks produced from documents")
        
        return splits
    
    def process_documents(self, paths: List[Path]) -> tuple[List[Document], List[str]]:
        """
        Load and split documents in one operation
        
        Args:
            paths: List of document paths
            
        Returns:
            Tuple of (split documents, list of errors)
        """
        documents, errors = self.load_documents(paths)
        
        if not documents:
            return [], errors
        
        try:
            splits = self.split_documents(documents)
            return splits, errors
        except ValueError as e:
            errors.append(str(e))
            return [], errors
    
    def update_settings(self, chunk_size: int, chunk_overlap: int):
        """
        Update chunking settings
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
