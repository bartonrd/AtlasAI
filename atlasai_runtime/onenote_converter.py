"""
OneNote to PDF converter.
Converts .one files to PDF format for better text extraction and processing.

This module provides a pure Python solution for converting OneNote (.one) files
to PDF format without requiring Windows COM automation or OneNote installation.
It uses pyOneNote for parsing and reportlab for PDF generation.
"""

import os
import shutil
import logging
from typing import List, Dict, Any
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Configuration constants
MAX_PROPERTIES = 50  # Maximum number of properties to include in PDF
MAX_EMBEDDED_FILES = 20  # Maximum number of embedded files to list in PDF
MAX_PROPERTY_VALUE_LENGTH = 500  # Maximum length for property values in debug output
MIN_TEXT_LENGTH = 2  # Minimum length for text to be considered content

# Property keys to skip when extracting text content
SKIP_PROPERTY_PATTERNS = [
    'guid', 'id', 'color', 'style', 'format', 'index',
    'offset', 'time', 'date', 'path', 'url', 'reference'
]


def convert_onenote_to_pdf(one_file_path: str, output_pdf_path: str, verbose: bool = False) -> bool:
    """
    Convert a OneNote file to PDF format using pure Python.
    
    This function provides an automated alternative to manually opening OneNote
    and saving as PDF. It uses pyOneNote to parse the .one file structure and
    reportlab to generate a searchable PDF document.
    
    Args:
        one_file_path: Path to the .one file
        output_pdf_path: Path where the PDF should be saved
        verbose: If True, log detailed conversion information
        
    Returns:
        True if conversion was successful, False otherwise
        
    Example:
        >>> convert_onenote_to_pdf("notes.one", "notes.pdf")
        Successfully converted notes.one to PDF
        True
    """
    if verbose:
        logger.info(f"Starting conversion: {one_file_path} -> {output_pdf_path}")
    
    try:
        # Import dependencies
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        # Note: The class is named 'OneDocment' (not 'OneDocument') in the pyOneNote library
        from pyOneNote.OneDocument import OneDocment
        
        # Validate input file exists
        if not os.path.exists(one_file_path):
            logger.error(f"Input file not found: {one_file_path}")
            return False
            
        # Create output directory if needed
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Parse the OneNote file
        if verbose:
            logger.info(f"Parsing OneNote file: {one_file_path}")
        
        with open(one_file_path, 'rb') as f:
            one_file = OneDocment(f)
        
        # Extract content
        content = _extract_content_from_onenote(one_file, one_file_path, verbose=verbose)
        
        if verbose:
            logger.info(f"Extracted {len(content)} content lines")
        
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
            alignment=TA_CENTER,
        )
        title = f"OneNote Document: {os.path.basename(one_file_path)}"
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Subtitle with conversion info
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=8,
            textColor='grey',
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        story.append(Paragraph("Converted from OneNote format using Python", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Content
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor='darkblue',
            spaceAfter=8,
        )
        
        for line in content:
            if line.strip():
                # Detect section headers (semantic headers like TITLE: or CONTENT:)
                # Exclude decorative separators (lines starting with = or -)
                is_semantic_header = (
                    line.strip().startswith('TITLE:') or 
                    line.strip().startswith('CONTENT:') or
                    line.strip().startswith('Author:') or
                    line.strip().startswith('Subject:')
                )
                is_separator = line.strip()[0] in ['=', '-'] if line.strip() else False
                
                # Escape special XML characters for ReportLab
                line_escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                if is_semantic_header and not is_separator:
                    # Use heading style for semantic headers
                    story.append(Paragraph(line_escaped, heading_style))
                else:
                    # Use normal style for content and separators
                    story.append(Paragraph(line_escaped, normal_style))
                    
                story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        if verbose:
            logger.info("Building PDF document...")
            
        doc.build(story)
        
        logger.info(f"Successfully converted {os.path.basename(one_file_path)} to PDF")
        return True
        
    except ImportError as e:
        logger.error(f"Missing required library for conversion: {e}")
        logger.error("Install with: pip install pyOneNote reportlab")
        return False
    except Exception as e:
        logger.error(f"Error converting {one_file_path} to PDF: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False


def _has_text_hint(key: str) -> bool:
    """Check if a property key might contain text content."""
    text_hints = ['text', 'string', 'data', 'content', 'title', 'author']
    key_lower = key.lower()
    return any(hint in key_lower for hint in text_hints)


def _is_hex_string(s: str) -> bool:
    """Check if a string is a hex-encoded string (and likely contains readable text)."""
    if len(s) < 20 or len(s) % 2 != 0:  # At least 10 chars of text when decoded
        return False
    # Must be all hex characters
    return all(c in '0123456789abcdefABCDEF' for c in s)


def _decode_hex_if_needed(s: str) -> str:
    """Decode hex string to text if it's a hex-encoded string."""
    if _is_hex_string(s):
        try:
            decoded = bytes.fromhex(s).decode('utf-8', errors='ignore')
            # Only return decoded if it's mostly readable
            if decoded and _is_readable_text(decoded):
                return decoded.strip()
        except (ValueError, UnicodeDecodeError):
            pass
    return s


def _is_readable_text(s: str) -> bool:
    """Check if a string contains mostly readable text."""
    if not s or len(s) < 3:
        return False
    
    # Check for byte string representations
    if s.startswith("b'") or s.startswith('b"'):
        return False
    
    # Count printable characters (excluding only null bytes)
    printable_count = sum(1 for c in s if c.isprintable() and c != '\x00')
    total_count = len(s)
    
    # String should be at least 50% printable
    if total_count > 0 and printable_count / total_count < 0.5:
        return False
    
    # Check for repetitive characters (like "nnnnnnnnn")
    if len(s) > 10:
        # Count most common character
        from collections import Counter
        char_counts = Counter(s)
        most_common_char, most_common_count = char_counts.most_common(1)[0]
        # If one character makes up more than 80% of the string, it's probably garbage
        if most_common_count / len(s) > 0.8:
            return False
    
    return True


def _extract_content_from_onenote(one_file, file_path: str, verbose: bool = False) -> List[str]:
    """
    Extract text content from a OneNote file.
    
    This function extracts various types of content from OneNote files including:
    - Document titles and metadata
    - Text content from properties
    - Author information
    - Embedded file information
    
    Args:
        one_file: Parsed OneNote document
        file_path: Original file path
        verbose: If True, log detailed extraction information
        
    Returns:
        List of text lines
    """
    content_lines = []
    
    # Add file info
    content_lines.append(f"Source: {os.path.basename(file_path)}")
    content_lines.append("")
    
    # Extract text content from properties
    text_content = []
    text_content_set = set()  # Track text to avoid duplicates with O(1) lookup
    title_found = False
    authors = set()  # Track authors to avoid duplicates
    metadata = {}
    
    try:
        properties = one_file.get_properties()
        if properties:
            if verbose:
                logger.info(f"Processing {len(properties)} properties from OneNote file")
            
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
                                if not any(skip in key.lower() for skip in SKIP_PROPERTY_PATTERNS):
                                    # Decode hex if needed
                                    text = _decode_hex_if_needed(value.strip())
                                    
                                    # Only add if it looks like real content
                                    if len(text) > MIN_TEXT_LENGTH and not text.isdigit() and _is_readable_text(text):
                                        # Avoid adding the same text multiple times (O(1) lookup)
                                        if text not in text_content_set:
                                            text_content.append(text)
                                            text_content_set.add(text)
    except Exception as e:
        content_lines.append(f"Error extracting text content: {e}")
        if verbose:
            logger.exception("Detailed extraction error:")
    
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
        if verbose:
            logger.info(f"Found {len(text_content)} unique text items")
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



def copy_onenote_files_locally(
    source_files: List[str],
    local_copy_dir: str,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Create local copies of OneNote files for non-destructive processing.
    
    This function copies .one files to a local directory, ensuring the original
    files remain untouched during conversion processing.
    
    Args:
        source_files: List of paths to source .one files
        local_copy_dir: Directory where local copies will be stored
        verbose: If True, log detailed copy information
        
    Returns:
        Dictionary mapping original file paths to local copy paths
        
    Example:
        >>> files = ["/network/notes1.one", "/network/notes2.one"]
        >>> copies = copy_onenote_files_locally(files, "local_copies/")
        >>> print(f"Created {len(copies)} local copies")
    """
    # Create local copy directory
    os.makedirs(local_copy_dir, exist_ok=True)
    
    copy_mapping = {}
    
    if verbose:
        logger.info(f"Creating local copies of {len(source_files)} file(s) in: {local_copy_dir}")
    
    for source_path in source_files:
        if not os.path.exists(source_path):
            logger.warning(f"Source file not found, skipping: {source_path}")
            continue
        
        try:
            # Generate local copy filename
            basename = os.path.basename(source_path)
            local_copy_path = os.path.join(local_copy_dir, basename)
            
            # Copy the file
            if verbose:
                logger.info(f"Copying: {basename}")
            
            shutil.copy2(source_path, local_copy_path)
            copy_mapping[source_path] = local_copy_path
            
            if verbose:
                logger.info(f"  -> Created local copy: {local_copy_path}")
                
        except Exception as e:
            logger.error(f"Error copying {source_path}: {e}")
            continue
    
    logger.info(f"Created {len(copy_mapping)} local cop{'y' if len(copy_mapping) == 1 else 'ies'}")
    
    return copy_mapping


def batch_convert_onenote_to_pdf(
    one_files: List[str],
    output_dir: str,
    verbose: bool = False,
    skip_existing: bool = False
) -> Dict[str, bool]:
    """
    Convert multiple OneNote files to PDF format.
    
    This is a convenience function for converting multiple OneNote files
    in a single operation without requiring Windows COM automation.
    
    Args:
        one_files: List of paths to .one files
        output_dir: Directory to save PDF files
        verbose: If True, log detailed conversion information
        skip_existing: If True, skip files where PDF already exists
        
    Returns:
        Dictionary mapping input file paths to success status
        
    Example:
        >>> files = ["notes1.one", "notes2.one"]
        >>> results = batch_convert_onenote_to_pdf(files, "output/")
        >>> print(f"Converted {sum(results.values())} of {len(files)} files")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    if verbose:
        logger.info(f"Starting batch conversion of {len(one_files)} files")
    
    for one_file_path in one_files:
        if not os.path.exists(one_file_path):
            logger.warning(f"File not found, skipping: {one_file_path}")
            results[one_file_path] = False
            continue
        
        # Generate output filename
        basename = os.path.splitext(os.path.basename(one_file_path))[0]
        output_pdf_path = os.path.join(output_dir, f"{basename}.pdf")
        
        # Skip if output exists and skip_existing is True
        if skip_existing and os.path.exists(output_pdf_path):
            if verbose:
                logger.info(f"Skipping existing file: {output_pdf_path}")
            results[one_file_path] = True
            continue
        
        # Convert the file
        success = convert_onenote_to_pdf(one_file_path, output_pdf_path, verbose=verbose)
        results[one_file_path] = success
    
    success_count = sum(results.values())
    if verbose:
        logger.info(f"Batch conversion complete: {success_count}/{len(one_files)} successful")
    
    return results


def convert_onenote_directory(
    source_dir: str,
    output_dir: str,
    overwrite: bool = True,
    verbose: bool = False,
    use_local_copies: bool = False,
    local_copy_dir: str = None
) -> int:
    """
    Convert all OneNote files in a directory to PDFs.
    
    This function provides an automated way to convert entire directories of
    OneNote files without manual intervention or Windows COM automation.
    
    Args:
        source_dir: Directory containing .one files
        output_dir: Directory to save PDF files
        overwrite: If True, clear output directory first and overwrite all files.
                   If False, keep existing files and skip conversion if PDF exists.
        verbose: If True, log detailed conversion information
        use_local_copies: If True, create local copies before conversion (non-destructive)
        local_copy_dir: Directory for local copies. If None and use_local_copies=True,
                        uses a subdirectory within output_dir
        
    Returns:
        Number of files successfully converted
        
    Example:
        >>> # Standard conversion
        >>> count = convert_onenote_directory("onenote_files/", "pdfs/")
        >>> 
        >>> # Non-destructive conversion with local copies
        >>> count = convert_onenote_directory(
        ...     "onenote_files/", "pdfs/", 
        ...     use_local_copies=True, 
        ...     local_copy_dir="local_copies/"
        ... )
    """
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If overwrite is True, clear the output directory first
    if overwrite and os.path.exists(output_dir):
        if verbose:
            logger.info(f"Clearing output directory: {output_dir}")
        try:
            # Only clear if directory has content
            if os.listdir(output_dir):
                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error clearing output directory: {e}")
            return 0
    
    # Find all .one files
    source_files = []
    try:
        for filename in os.listdir(source_dir):
            if filename.lower().endswith('.one'):
                source_files.append(os.path.join(source_dir, filename))
    except Exception as e:
        logger.error(f"Error listing files in {source_dir}: {e}")
        return 0
    
    if not source_files:
        logger.warning(f"No .one files found in {source_dir}")
        return 0
    
    logger.info(f"Found {len(source_files)} OneNote file(s) to convert")
    
    # Determine which files to convert
    files_to_convert = source_files
    
    # If using local copies, create them first (non-destructive approach)
    if use_local_copies:
        # Determine local copy directory
        if local_copy_dir is None:
            local_copy_dir = os.path.join(output_dir, "..", "onenote_copies")
            local_copy_dir = os.path.abspath(local_copy_dir)
        
        logger.info(f"[NON-DESTRUCTIVE MODE] Creating local copies before conversion")
        logger.info(f"Local copy directory: {local_copy_dir}")
        
        # Create local copies
        copy_mapping = copy_onenote_files_locally(
            source_files,
            local_copy_dir,
            verbose=verbose
        )
        
        if not copy_mapping:
            logger.error("Failed to create any local copies")
            return 0
        
        # Convert from local copies instead of originals
        files_to_convert = list(copy_mapping.values())
        logger.info(f"[NON-DESTRUCTIVE MODE] Will process {len(files_to_convert)} local copies")
    
    # Use batch conversion with skip_existing controlled by overwrite parameter
    # When overwrite=False, skip existing files
    results = batch_convert_onenote_to_pdf(
        files_to_convert,
        output_dir,
        verbose=verbose,
        skip_existing=not overwrite  # If overwrite=False, skip existing files
    )
    
    converted_count = sum(results.values())
    logger.info(f"Successfully converted {converted_count}/{len(files_to_convert)} file(s)")
    
    if use_local_copies:
        logger.info(f"[NON-DESTRUCTIVE MODE] Original files remain untouched in: {source_dir}")
        logger.info(f"[NON-DESTRUCTIVE MODE] Local copies stored in: {local_copy_dir}")
    
    return converted_count


# Backward compatibility - keep old function signature
def convert_onenote_directory_legacy(
    source_dir: str,
    output_dir: str,
    overwrite: bool = True
) -> int:
    """
    Legacy function for backward compatibility.
    Use convert_onenote_directory() instead.
    
    Convert all OneNote files in a directory to PDFs.
    
    Args:
        source_dir: Directory containing .one files
        output_dir: Directory to save PDF files
        overwrite: If True, overwrite existing PDFs
        
    Returns:
        Number of files successfully converted
    """
    return convert_onenote_directory(source_dir, output_dir, overwrite, verbose=False)


def get_conversion_info() -> Dict[str, Any]:
    """
    Get information about the OneNote to PDF conversion capabilities.
    
    Returns:
        Dictionary with conversion information including:
        - supported_formats: List of supported input formats
        - requires_windows: Whether Windows is required
        - requires_onenote: Whether OneNote app is required
        - method: Description of conversion method
    """
    return {
        "supported_formats": [".one"],
        "output_format": ".pdf",
        "requires_windows": False,
        "requires_onenote": False,
        "requires_com_automation": False,
        "method": "Pure Python using pyOneNote and reportlab",
        "advantages": [
            "No OneNote installation required",
            "Cross-platform compatible",
            "Fully automated",
            "No Windows COM automation needed",
            "Can run on Linux/Mac",
        ],
        "limitations": [
            "Text extraction depends on OneNote file structure",
            "May not preserve complex formatting",
            "Primarily extracts metadata and text properties",
        ]
    }
