"""
OneNote to PDF converter.
Converts .one files to PDF format for better text extraction and processing.
"""

import os
import shutil
from typing import List
from pathlib import Path

# Configuration constants
MAX_PROPERTIES = 50  # Maximum number of properties to include in PDF
MAX_EMBEDDED_FILES = 20  # Maximum number of embedded files to list in PDF
MAX_PROPERTY_VALUE_LENGTH = 500  # Maximum length for property values in debug output


def convert_onenote_to_pdf(one_file_path: str, output_pdf_path: str) -> bool:
    """
    Convert a OneNote file to PDF format.
    
    Args:
        one_file_path: Path to the .one file
        output_pdf_path: Path where the PDF should be saved
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Import dependencies
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT
        # Note: The class is named 'OneDocment' (not 'OneDocument') in the pyOneNote library
        from pyOneNote.OneDocument import OneDocment
        
        # Parse the OneNote file
        with open(one_file_path, 'rb') as f:
            one_file = OneDocment(f)
        
        # Extract content
        content = _extract_content_from_onenote(one_file, one_file_path)
        
        # Create PDF
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        # Build PDF content
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='black',
            spaceAfter=12,
        )
        title = f"OneNote Document: {os.path.basename(one_file_path)}"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Content
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
        
        for line in content:
            if line.strip():
                # Escape special XML characters for ReportLab
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        
        print(f"Successfully converted {os.path.basename(one_file_path)} to PDF")
        return True
        
    except Exception as e:
        print(f"Error converting {one_file_path} to PDF: {e}")
        return False


def _has_text_hint(key: str) -> bool:
    """Check if a property key might contain text content."""
    text_hints = ['text', 'string', 'data', 'content', 'title', 'author']
    key_lower = key.lower()
    return any(hint in key_lower for hint in text_hints)


def _extract_content_from_onenote(one_file, file_path: str) -> List[str]:
    """
    Extract text content from a OneNote file.
    
    Args:
        one_file: Parsed OneNote document
        file_path: Original file path
        
    Returns:
        List of text lines
    """
    content_lines = []
    
    # Add file info
    content_lines.append(f"Source: {os.path.basename(file_path)}")
    content_lines.append("")
    
    # Extract text content from properties
    text_content = []
    title_found = False
    authors = set()  # Track authors to avoid duplicates
    metadata = {}
    
    try:
        properties = one_file.get_properties()
        if properties:
            for prop in properties:
                if isinstance(prop, dict):
                    prop_type = prop.get('type', '')
                    val = prop.get('val', {})
                    
                    if isinstance(val, dict):
                        # Extract title strings (show only once)
                        if 'CachedTitleString' in val or 'CachedTitleStringFromPage' in val:
                            title = val.get('CachedTitleString') or val.get('CachedTitleStringFromPage')
                            if title and isinstance(title, str) and title.strip():
                                if not title_found:
                                    content_lines.append("=" * 60)
                                    content_lines.append(f"TITLE: {title.strip()}")
                                    content_lines.append("=" * 60)
                                    content_lines.append("")
                                    title_found = True
                        
                        # Extract all string values that might contain text content
                        # Exclude known metadata fields and process them separately
                        for key, value in val.items():
                            # Handle metadata fields specially
                            if key in ['Author', 'LastModifiedBy', 'CreatedBy']:
                                if value and isinstance(value, str) and value.strip():
                                    if key == 'Author':
                                        authors.add(value.strip())
                                    else:
                                        metadata[key] = value.strip()
                            elif key in ['Subject', 'Keywords', 'Category']:
                                if value and isinstance(value, str) and value.strip():
                                    metadata[key] = value.strip()
                            # Extract actual text content from string properties
                            elif isinstance(value, str) and value.strip():
                                # Skip properties that are clearly not content
                                if not any(skip in key.lower() for skip in [
                                    'guid', 'id', 'color', 'style', 'format', 'index', 
                                    'offset', 'time', 'date', 'path', 'url', 'reference'
                                ]):
                                    # This might be actual text content
                                    text = value.strip()
                                    # Only add if it looks like real content (not just a single word or ID)
                                    if len(text) > 2 and not text.isdigit():
                                        # Avoid adding the same text multiple times
                                        if text not in text_content:
                                            text_content.append(text)
    except Exception as e:
        content_lines.append(f"Error extracting text content: {e}")
    
    # Add metadata (authors only once)
    if authors:
        author_list = sorted(list(authors))
        if len(author_list) == 1:
            content_lines.append(f"Author: {author_list[0]}")
        else:
            content_lines.append(f"Authors: {', '.join(author_list)}")
    
    for key, value in sorted(metadata.items()):
        content_lines.append(f"{key}: {value}")
    
    if authors or metadata:
        content_lines.append("")
    
    # Add extracted text content
    if text_content:
        content_lines.append("CONTENT:")
        content_lines.append("-" * 60)
        for text in text_content:
            content_lines.append(text)
            content_lines.append("")
    
    # If no text was extracted, show debug info
    if not text_content and not title_found:
        content_lines.append("")
        content_lines.append("Note: No text content found. Extracting available properties for debugging...")
        content_lines.append("")
        try:
            properties = one_file.get_properties()
            if properties:
                content_lines.append(f"Available Properties ({len(properties)} items):")
                content_lines.append("-" * 50)
                
                for prop in properties[:MAX_PROPERTIES]:
                    if isinstance(prop, dict):
                        prop_type = prop.get('type', 'Unknown')
                        val = prop.get('val', {})
                        
                        if isinstance(val, dict):
                            # List all keys that might contain useful information
                            text_keys = [k for k in val.keys() if _has_text_hint(k)]
                            
                            if text_keys:
                                content_lines.append(f"Property Type: {prop_type}")
                                for key in text_keys:
                                    value = val[key]
                                    if value:
                                        value_str = str(value)[:MAX_PROPERTY_VALUE_LENGTH]
                                        content_lines.append(f"  {key}: {value_str}")
                                content_lines.append("")
        except Exception as e:
            content_lines.append(f"Error extracting metadata: {e}")
    
    # Extract embedded files info
    try:
        files = one_file.get_files()
        if files:
            content_lines.append("")
            content_lines.append(f"Embedded Files ({len(files)} items):")
            content_lines.append("-" * 50)
            
            for file_guid, file_info in list(files.items())[:MAX_EMBEDDED_FILES]:
                if isinstance(file_info, dict):
                    extension = file_info.get('extension', 'Unknown')
                    identity = file_info.get('identity', 'Unknown')
                    content_size = len(file_info.get('content', b''))
                    content_lines.append(f"- File: {file_guid}")
                    content_lines.append(f"  Extension: {extension}")
                    content_lines.append(f"  Size: {content_size} bytes")
                    content_lines.append("")
    except Exception as e:
        content_lines.append(f"Error extracting files: {e}")
    
    return content_lines



def convert_onenote_directory(
    source_dir: str,
    output_dir: str,
    overwrite: bool = True
) -> int:
    """
    Convert all OneNote files in a directory to PDFs.
    
    Args:
        source_dir: Directory containing .one files
        output_dir: Directory to save PDF files
        overwrite: If True, overwrite existing PDFs
        
    Returns:
        Number of files successfully converted
    """
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If overwrite is True, clear the output directory
    if overwrite and os.path.exists(output_dir):
        print(f"Clearing output directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error clearing output directory: {e}")
            return 0
    
    # Find all .one files
    one_files = []
    try:
        for filename in os.listdir(source_dir):
            if filename.lower().endswith('.one'):
                one_files.append(filename)
    except Exception as e:
        print(f"Error listing files in {source_dir}: {e}")
        return 0
    
    if not one_files:
        print(f"No .one files found in {source_dir}")
        return 0
    
    print(f"Found {len(one_files)} OneNote file(s) to convert")
    
    # Convert each file
    converted_count = 0
    for filename in one_files:
        source_path = os.path.join(source_dir, filename)
        pdf_filename = os.path.splitext(filename)[0] + '.pdf'
        output_path = os.path.join(output_dir, pdf_filename)
        
        print(f"Converting: {filename} -> {pdf_filename}")
        if convert_onenote_to_pdf(source_path, output_path):
            converted_count += 1
    
    print(f"Successfully converted {converted_count}/{len(one_files)} file(s)")
    return converted_count
