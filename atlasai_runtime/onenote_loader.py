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
            # Import pyOneNote modules here to avoid import errors if not installed
            from pyOneNote.OneDocument import OneDocment
            
            # Parse the OneNote file
            with open(self.file_path, 'rb') as f:
                one_file = OneDocment(f)
            
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
            one_file: Parsed OneNote file object (OneDocument instance)
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        # Try to get file metadata and properties
        try:
            # Get properties from the OneNote file
            properties = one_file.get_properties()
            if properties:
                text_parts.append(f"OneNote document with {len(properties)} properties")
                
                # Extract text from properties
                for prop in properties:
                    if isinstance(prop, dict):
                        # Look for text-related properties
                        prop_type = prop.get('type', '')
                        if prop_type and 'jcid' in prop_type.lower():
                            text_parts.append(f"- {prop_type}")
                        
                        # Try to extract meaningful text values
                        val = prop.get('val', {})
                        if isinstance(val, dict):
                            # Look for text content in common keys
                            for key in ['TextContent', 'Title', 'Name', 'Author', 'Subject']:
                                if key in val and val[key]:
                                    text_parts.append(f"{key}: {val[key]}")
        except Exception as e:
            print(f"Debug: Error extracting properties: {e}")
        
        # Try to get embedded files information
        try:
            files = one_file.get_files()
            if files:
                text_parts.append(f"\nEmbedded files ({len(files)}):")
                for file_info in files[:10]:  # Limit to first 10
                    if isinstance(file_info, dict):
                        file_name = file_info.get('file_name', 'Unknown')
                        text_parts.append(f"- {file_name}")
        except Exception as e:
            print(f"Debug: Error extracting files: {e}")
        
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
