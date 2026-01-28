# OneNote to PDF Conversion Guide

## Overview

AtlasAI provides a **pure Python solution** for converting OneNote (.one) files to PDF format. This solution:

- ✅ **No OneNote installation required**
- ✅ **Cross-platform compatible** (Windows, Linux, macOS)
- ✅ **Fully automated** - no manual steps needed
- ✅ **No COM automation** - pure Python implementation
- ✅ **Batch processing** - convert multiple files at once
- ✅ **Non-destructive mode** - work on local copies, preserve originals

## Important: Non-Destructive Mode

The conversion now supports **non-destructive mode** which:
- Creates local copies of OneNote files before processing
- Leaves original .one files completely untouched
- Stores local copies in the `documents/onenote_copies/` directory
- Processes only the local copies for conversion

This ensures your original OneNote files are never modified or damaged during the conversion process.

## Why This Approach is Better

### Traditional Approach (Manual/COM Automation)
The traditional way to convert OneNote files to PDF involves:
1. Opening OneNote application
2. Manually opening each .one file
3. Using File > Export > PDF
4. Saving each file individually

Or using Windows COM automation:
- Requires Windows OS
- Requires OneNote installation
- Uses `win32com` or `comtypes` libraries
- Prone to COM object lifetime issues
- Not portable to other platforms

### Our Approach (Pure Python)
Our solution uses:
- `pyOneNote` - Pure Python library for parsing .one files
- `reportlab` - Pure Python library for generating PDFs
- No external applications needed
- Works on any platform
- **Non-destructive mode** protects original files

### Known Limitations

The pure Python approach has some limitations:
- **Special characters**: May not render complex special characters perfectly
- **Screenshots/Images**: Does not extract embedded images (pyOneNote limitation)
- **Formatting**: Complex formatting may not be preserved

The conversion extracts text content, metadata, and document structure, which is sufficient for searchable PDF documents and RAG processing.

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

The key packages for OneNote conversion are:
- `pyOneNote>=0.0.2` - For parsing OneNote files
- `reportlab>=4.0.0` - For generating PDFs

## Usage

### 1. Command-Line Interface

The easiest way to convert files is using the `convert_onenote.py` script:

#### Convert a Single File

```bash
python convert_onenote.py input.one output.pdf
```

#### Convert an Entire Directory

```bash
python convert_onenote.py onenote_files/ pdf_output/ --directory
```

#### Non-Destructive Conversion (Recommended)

Use this mode to create local copies before conversion, ensuring original files are never modified:

```bash
# Non-destructive mode with local copies
python convert_onenote.py onenote_files/ pdf_output/ --directory --use-local-copies

# Specify custom local copy directory
python convert_onenote.py onenote_files/ pdf_output/ --directory --use-local-copies --local-copy-dir my_copies/
```

#### Verbose Mode (See Detailed Progress)

```bash
python convert_onenote.py input.one output.pdf --verbose
```

#### Show Conversion Capabilities

```bash
python convert_onenote.py --info
```

### 2. Programmatic Usage

You can also use the conversion functions directly in your Python code:

#### Convert a Single File

```python
from atlasai_runtime.onenote_converter import convert_onenote_to_pdf

# Basic conversion
success = convert_onenote_to_pdf("notes.one", "notes.pdf")

# With verbose logging
success = convert_onenote_to_pdf("notes.one", "notes.pdf", verbose=True)
```

#### Batch Convert Multiple Files

```python
from atlasai_runtime.onenote_converter import batch_convert_onenote_to_pdf

files = ["notes1.one", "notes2.one", "notes3.one"]
results = batch_convert_onenote_to_pdf(
    files,
    output_dir="pdf_output/",
    verbose=True,
    skip_existing=False
)

# Check results
for file, success in results.items():
    if success:
        print(f"✓ {file} converted successfully")
    else:
        print(f"✗ {file} conversion failed")
```

#### Convert Entire Directory

```python
from atlasai_runtime.onenote_converter import convert_onenote_directory

# Standard conversion
count = convert_onenote_directory(
    source_dir="onenote_files/",
    output_dir="pdf_output/",
    overwrite=True,
    verbose=True
)

print(f"Converted {count} files")
```

#### Non-Destructive Conversion (Recommended)

```python
from atlasai_runtime.onenote_converter import convert_onenote_directory

# Non-destructive mode with local copies
count = convert_onenote_directory(
    source_dir="onenote_files/",
    output_dir="pdf_output/",
    overwrite=True,
    verbose=True,
    use_local_copies=True,  # Enable non-destructive mode
    local_copy_dir="local_copies/"  # Optional: specify copy location
)

print(f"Converted {count} files (originals untouched)")
```

#### Copy OneNote Files Locally

```python
from atlasai_runtime.onenote_converter import copy_onenote_files_locally

# Create local copies
files = ["/network/notes1.one", "/network/notes2.one"]
copy_mapping = copy_onenote_files_locally(
    files,
    local_copy_dir="local_copies/",
    verbose=True
)

# copy_mapping contains: {original_path: local_copy_path}
for original, copy in copy_mapping.items():
    print(f"Copied: {original} -> {copy}")
```

#### Get Conversion Information

```python
from atlasai_runtime.onenote_converter import get_conversion_info

info = get_conversion_info()
print(f"Method: {info['method']}")
print(f"Requires Windows: {info['requires_windows']}")
print(f"Requires OneNote: {info['requires_onenote']}")
```

### 3. Automatic Conversion (Built into AtlasAI)

AtlasAI automatically converts OneNote files during startup:

1. Configure the OneNote runbook path:
   ```bash
   export ATLASAI_ONENOTE_RUNBOOK_PATH="/path/to/onenote/files"
   ```

2. Run AtlasAI:
   ```bash
   dotnet run
   ```

3. The application will automatically:
   - Scan the configured path for .one files
   - Convert all found files to PDF
   - Save PDFs to `documents/runbook/`
   - Load them into the RAG system

## What Gets Extracted

The conversion extracts:
- **Document titles** - Page titles and section names
- **Metadata** - Authors, creation dates, modification dates
- **Text content** - All text properties in the OneNote structure
- **Embedded files** - Information about attached files (names, types, sizes)

## Limitations

Due to the complexity of the OneNote file format:
- Text extraction depends on how OneNote stores content internally
- Complex formatting (colors, fonts, layouts) may not be preserved
- Handwritten notes and drawings are not extracted
- Some OneNote features may not have direct PDF equivalents

However, the pure Python approach ensures:
- Cross-platform compatibility
- No external dependencies
- Fully automated processing
- Consistent results across environments

## Troubleshooting

### ImportError: No module named 'pyOneNote'

Install the required package:
```bash
pip install pyOneNote
```

### ImportError: No module named 'reportlab'

Install the required package:
```bash
pip install reportlab
```

### "No extractable text content found"

This means the OneNote file structure doesn't contain easily extractable text. This can happen with:
- OneNote files that primarily contain images
- Files with handwritten notes
- Corrupted or incomplete .one files

The conversion will still create a PDF with metadata and file information.

### Permission Errors

Ensure you have:
- Read permission on the input .one file
- Write permission in the output directory

## Examples

### Example 1: Convert Meeting Notes

```bash
python convert_onenote.py "Meeting Notes.one" "Meeting Notes.pdf" --verbose
```

### Example 2: Batch Convert Project Documentation

```bash
python convert_onenote.py "Project Docs/" "PDF Docs/" --directory
```

### Example 3: Script for Daily Backup

```python
#!/usr/bin/env python3
import os
from datetime import datetime
from atlasai_runtime.onenote_converter import convert_onenote_directory

# Configuration
SOURCE_DIR = "/path/to/onenote/files"
BACKUP_DIR = f"/path/to/backups/{datetime.now().strftime('%Y-%m-%d')}"

# Convert all OneNote files to PDF backup
os.makedirs(BACKUP_DIR, exist_ok=True)
count = convert_onenote_directory(SOURCE_DIR, BACKUP_DIR, verbose=True)
print(f"Backed up {count} files to {BACKUP_DIR}")
```

## Performance

Conversion time depends on:
- **File size** - Larger files take longer
- **Content complexity** - More properties = longer processing
- **System resources** - CPU and disk I/O

Typical performance:
- Small files (<1MB): 1-2 seconds
- Medium files (1-5MB): 2-5 seconds
- Large files (>5MB): 5-15 seconds

## Support

For issues or questions:
1. Check this guide for common solutions
2. Review the `--info` output for capability information
3. Use `--verbose` flag to see detailed logging
4. Check the AtlasAI GitHub repository for updates

## Summary

The pure Python OneNote to PDF conversion provides:
- **Automated conversion** without manual intervention
- **Cross-platform support** on Windows, Linux, and macOS
- **No external applications** required
- **Batch processing** for multiple files
- **Easy integration** into Python scripts and workflows

This is a much better approach than manual conversion or Windows COM automation!
