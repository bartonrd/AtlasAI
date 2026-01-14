"""
Custom OneNote document loader for LangChain.
"""

import os
import tempfile
from typing import List, Iterator
from pathlib import Path

from langchain_core.documents import Document


class OneNoteLoader:
    """
    Load OneNote (.one) files and convert them to LangChain Documents.
    
    This loader uses pyOneNote to extract content from OneNote files.
    Note: pyOneNote focuses on extracting embedded files, so text extraction
    is limited. This loader attempts to extract whatever text is available.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the OneNote loader.
        
        Args:
            file_path: Path to the .one file
        """
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """
        Load the OneNote file and return documents.
        
        Returns:
            List of Document objects
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        try:
            # Import pyOneNote here to avoid import errors if not installed
            import pyOneNote
            
            # Parse the OneNote file
            with open(self.file_path, 'rb') as f:
                one_file = pyOneNote.OneNoteFile(f.read())
            
            # Extract text content
            text_content = self._extract_text(one_file)
            
            # Create a document with the extracted content
            metadata = {
                "source": self.file_path,
                "file_type": "onenote",
            }
            
            # If we have text, create a document
            if text_content:
                doc = Document(page_content=text_content, metadata=metadata)
                return [doc]
            else:
                # If no text found, create a minimal document noting this
                doc = Document(
                    page_content=f"OneNote file: {os.path.basename(self.file_path)} (no extractable text content found)",
                    metadata=metadata
                )
                return [doc]
                
        except ImportError:
            raise ImportError(
                "pyOneNote is required to load OneNote files. "
                "Install it with: pip install pyOneNote"
            )
        except Exception as e:
            # If parsing fails, create a document noting the error
            print(f"Warning: Failed to parse OneNote file {self.file_path}: {e}")
            doc = Document(
                page_content=f"OneNote file: {os.path.basename(self.file_path)} (parsing error: {str(e)})",
                metadata={"source": self.file_path, "file_type": "onenote", "error": str(e)}
            )
            return [doc]
    
    def _extract_text(self, one_file) -> str:
        """
        Extract text content from a OneNote file.
        
        Args:
            one_file: Parsed OneNote file object
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        # Try to get the file GUID if available
        try:
            if hasattr(one_file, 'guid'):
                text_parts.append(f"OneNote File ID: {one_file.guid}")
        except:
            pass
        
        # Try to extract embedded files and their names (which might contain useful info)
        try:
            if hasattr(one_file, 'get_files'):
                files = one_file.get_files()
                if files:
                    text_parts.append("Embedded files:")
                    for file_info in files:
                        if hasattr(file_info, 'file_name'):
                            text_parts.append(f"- {file_info.file_name}")
        except:
            pass
        
        # Try to get any text data that might be available
        # Note: pyOneNote is primarily for embedded file extraction,
        # so text extraction is very limited
        try:
            if hasattr(one_file, 'data'):
                # Try to find readable text in the data
                # This is a best-effort approach
                data_str = str(one_file.data)
                # Look for readable ASCII text chunks (very basic heuristic)
                import re
                # Find sequences of printable characters
                text_chunks = re.findall(r'[a-zA-Z0-9\s\.,;:!?()-]{20,}', data_str)
                if text_chunks:
                    text_parts.extend(text_chunks[:10])  # Limit to first 10 chunks
        except:
            pass
        
        if text_parts:
            return "\n".join(text_parts)
        
        return ""
    
    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load documents (calls load()).
        
        Yields:
            Document objects
        """
        yield from self.load()
