"""
OneNote document loader for RAG system.

This module provides functionality to ingest OneNote .one documents
using the OneNote COM API on Windows.
"""

import os
import logging
import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Set

from langchain_core.documents import Document

# Lazy imports for Windows-specific modules
try:
    import win32com.client
    ONENOTE_AVAILABLE = True
except ImportError:
    ONENOTE_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)

# OneNote COM API constants
PF_HTML = 1
PF_MHTML = 2
HS_PAGES = 3


def ingest_onenote(
    runbook_root: Path,
    temp_dir: Optional[Path] = None,
    enable_onenote: bool = True
) -> List[Document]:
    """
    Ingest OneNote documents from a UNC path.

    Args:
        runbook_root: Root path to search for .one files (can be UNC path)
        temp_dir: Temporary directory for HTML exports (default: ./tmp_onenote)
        enable_onenote: Feature flag to enable/disable OneNote ingestion

    Returns:
        List of LangChain Document objects with page content and metadata.
        Returns empty list if OneNote is disabled or unavailable.
    """
    if not enable_onenote:
        logger.info("OneNote ingestion disabled via feature flag")
        return []

    if not ONENOTE_AVAILABLE:
        logger.warning(
            "OneNote COM API not available (pywin32 not installed). "
            "Skipping OneNote ingestion. Install pywin32 to enable this feature."
        )
        return []

    if not BS4_AVAILABLE:
        logger.warning(
            "BeautifulSoup not available. Skipping OneNote ingestion. "
            "Install beautifulsoup4 to enable this feature."
        )
        return []

    # Set up temp directory
    if temp_dir is None:
        temp_dir = Path("./tmp_onenote")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize OneNote COM object
        try:
            # Try early binding first (recommended for better type information)
            try:
                onenote = win32com.client.gencache.EnsureDispatch("OneNote.Application")
                logger.debug("OneNote COM initialized with early binding")
            except:
                # Fall back to late binding
                onenote = win32com.client.Dispatch("OneNote.Application")
                logger.debug("OneNote COM initialized with late binding")
        except Exception as e:
            logger.warning(f"Failed to initialize OneNote COM API: {e}. Skipping OneNote ingestion.")
            return []

        # Enumerate .one files
        one_files = _enumerate_one_files(runbook_root)
        if not one_files:
            logger.info(f"No .one files found in {runbook_root}")
            return []

        logger.info(f"Found {len(one_files)} .one files in {runbook_root}")

        # Try to open the parent folder as a notebook
        # This ensures OneNote knows about all the .one files in the directory
        try:
            onenote.OpenHierarchy(str(runbook_root), "", None, 0)
            logger.debug(f"Opened hierarchy for {runbook_root}")
        except Exception as e:
            logger.debug(f"Could not open hierarchy for parent folder: {e}")
        
        # Process each .one file
        documents = []
        processed_page_ids: Set[str] = set()

        for one_file in one_files:
            try:
                docs = _process_one_file(
                    onenote, one_file, temp_dir, processed_page_ids
                )
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to process {one_file}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully ingested {len(documents)} OneNote pages")

        # Clean up temp directory on success
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

        return documents

    except Exception as e:
        logger.error(f"Error during OneNote ingestion: {e}", exc_info=True)
        return []


def _enumerate_one_files(root_path: Path) -> List[Path]:
    """
    Enumerate all .one files recursively, skipping temp/lock files.

    Args:
        root_path: Root directory to search

    Returns:
        List of .one file paths
    """
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        return []

    try:
        one_files = []
        for one_file in root_path.rglob("*.one"):
            # Skip temp/lock files (typically start with ~$ or contain .cache)
            if one_file.name.startswith("~$") or ".cache" in one_file.name.lower():
                continue
            one_files.append(one_file)
        return one_files
    except Exception as e:
        logger.error(f"Error enumerating .one files from {root_path}: {e}")
        # Provide actionable message for UNC path issues
        if str(root_path).startswith("\\\\"):
            logger.error(
                f"UNC path access error. Please ensure:\n"
                f"  1. The network path {root_path} is accessible\n"
                f"  2. You have appropriate permissions to access the path\n"
                f"  3. The network share is mounted/available"
            )
        return []


def _process_one_file(
    onenote,
    one_file: Path,
    temp_dir: Path,
    processed_page_ids: Set[str]
) -> List[Document]:
    """
    Process a single .one file and extract all pages.

    Args:
        onenote: OneNote COM application object
        one_file: Path to .one file
        temp_dir: Temporary directory for exports
        processed_page_ids: Set of already processed page IDs (for deduplication)

    Returns:
        List of Document objects for this .one file
    """
    documents = []

    try:
        # Open the section/notebook first to ensure it's accessible
        # This is required for the COM API to work with the file
        try:
            onenote.OpenHierarchy(str(one_file), "", None, 0)
            logger.debug(f"OpenHierarchy succeeded for {one_file.name}")
        except Exception as e:
            logger.debug(f"OpenHierarchy failed for {one_file.name}: {e}")
            # Continue anyway - file might already be open
        
        # Get ALL notebooks hierarchy first, then find our section
        # Individual .one files are sections, not notebooks
        # We need to get the full hierarchy and find the section by path
        try:
            # Try calling GetHierarchy with empty string to get all notebooks
            all_hierarchy_xml = onenote.GetHierarchy("", HS_PAGES)
        except AttributeError as e:
            # GetHierarchy might not be available, try with win32com dynamic dispatch
            logger.error(f"GetHierarchy not available on OneNote COM object: {e}")
            logger.error(f"Available methods: {dir(onenote)}")
            raise
        
        all_root = ET.fromstring(all_hierarchy_xml)
        
        # Define namespace for OneNote XML
        ns = {"one": "http://schemas.microsoft.com/office/onenote/2013/onenote"}
        
        # Find the section that matches our .one file path
        target_path = str(one_file).lower()
        pages = []
        
        for section in all_root.findall(".//one:Section", ns):
            section_path = section.get("path", "")
            if section_path.lower() == target_path:
                # Found our section, get its pages
                pages = section.findall(".//one:Page", ns)
                logger.debug(f"Found {len(pages)} pages in section {one_file.name}")
                break

        # Parse the XML hierarchy
        root = ET.fromstring(all_hierarchy_xml)

        # Find all pages in the hierarchy
        for page in pages:
            page_id = page.get("ID")
            page_name = page.get("name", "Untitled")

            # Skip if already processed (deduplication)
            if page_id in processed_page_ids:
                logger.debug(f"Skipping duplicate page: {page_name} ({page_id})")
                continue

            try:
                # Export page to HTML
                html_path = temp_dir / f"{hashlib.sha256(page_id.encode()).hexdigest()}.html"
                onenote.Publish(page_id, str(html_path), PF_HTML, "")

                # Extract text from HTML
                if html_path.exists():
                    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                        html_content = f.read()

                    # Parse HTML and extract clean text
                    text = _extract_text_from_html(html_content)

                    if text.strip():
                        # Get section and notebook info from hierarchy
                        section = _get_parent_section(page, root, ns)
                        notebook = _get_parent_notebook(page, root, ns)

                        # Create Document with metadata
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": str(one_file),
                                "page": page_name,
                                "title": page_name,
                                "section": section,
                                "notebook": notebook,
                                "page_id": page_id,
                            }
                        )
                        documents.append(doc)
                        processed_page_ids.add(page_id)
                        logger.debug(f"Processed page: {page_name}")

                    # Clean up HTML file
                    try:
                        html_path.unlink()
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"Failed to process page {page_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to get hierarchy for {one_file}: {type(e).__name__}: {e}")

    return documents


def _extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content using BeautifulSoup.

    Args:
        html_content: HTML content as string

    Returns:
        Extracted plain text
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()

    # Get text with line breaks
    text = soup.get_text("\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text


def _get_parent_section(page_element, root, ns) -> str:
    """
    Get the section name for a page element.

    Args:
        page_element: Page XML element
        root: Root XML element
        ns: XML namespace dict

    Returns:
        Section name or "Unknown"
    """
    try:
        # Use XPath to find parent section by checking if page ID is in section's descendants
        page_id = page_element.get("ID")
        if page_id:
            for section in root.findall(".//one:Section", ns):
                # Check if any page in this section has matching ID
                for page in section.findall(".//one:Page[@ID]", ns):
                    if page.get("ID") == page_id:
                        return section.get("name", "Unknown")
    except Exception:
        pass
    return "Unknown"


def _get_parent_notebook(page_element, root, ns) -> str:
    """
    Get the notebook name for a page element.

    Args:
        page_element: Page XML element
        root: Root XML element
        ns: XML namespace dict

    Returns:
        Notebook name or "Unknown"
    """
    try:
        # Use XPath to find parent notebook by checking if page ID is in notebook's descendants
        page_id = page_element.get("ID")
        if page_id:
            for notebook in root.findall(".//one:Notebook", ns):
                # Check if any page in this notebook has matching ID
                for page in notebook.findall(".//one:Page[@ID]", ns):
                    if page.get("ID") == page_id:
                        return notebook.get("name", "Unknown")
    except Exception:
        pass
    return "Unknown"
