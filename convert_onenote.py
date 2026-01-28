#!/usr/bin/env python3
"""
Standalone script for converting OneNote files to PDF.

This script provides a command-line interface for converting OneNote (.one) files
to PDF format without requiring Windows COM automation or OneNote installation.

Usage:
    # Convert a single file
    python convert_onenote.py input.one output.pdf
    
    # Convert a directory
    python convert_onenote.py input_dir/ output_dir/ --directory
    
    # Convert with verbose logging
    python convert_onenote.py input.one output.pdf --verbose
    
    # Show conversion capabilities
    python convert_onenote.py --info
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlasai_runtime.onenote_converter import (
    convert_onenote_to_pdf,
    convert_onenote_directory,
    batch_convert_onenote_to_pdf,
    get_conversion_info
)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_info():
    """Print information about conversion capabilities."""
    info = get_conversion_info()
    
    print("\n=== OneNote to PDF Conversion Info ===\n")
    print(f"Supported input formats: {', '.join(info['supported_formats'])}")
    print(f"Output format: {info['output_format']}")
    print(f"Requires Windows: {info['requires_windows']}")
    print(f"Requires OneNote app: {info['requires_onenote']}")
    print(f"Requires COM automation: {info['requires_com_automation']}")
    print(f"Method: {info['method']}\n")
    
    print("Advantages:")
    for adv in info['advantages']:
        print(f"  ✓ {adv}")
    
    print("\nLimitations:")
    for lim in info['limitations']:
        print(f"  • {lim}")
    
    print("\nThis is a pure Python solution that works on any platform!")
    print()


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert OneNote files to PDF format without OneNote installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_onenote.py notes.one notes.pdf
  
  # Convert directory
  python convert_onenote.py onenote_files/ pdf_output/ --directory
  
  # Convert with verbose output
  python convert_onenote.py notes.one notes.pdf -v
  
  # Show conversion info
  python convert_onenote.py --info
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Input .one file or directory'
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output .pdf file or directory'
    )
    parser.add_argument(
        '-d', '--directory',
        action='store_true',
        help='Convert all .one files in input directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Skip existing files instead of overwriting'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show conversion capabilities and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle --info flag
    if args.info:
        print_info()
        return 0
    
    # Validate arguments
    if not args.input or not args.output:
        parser.print_help()
        print("\nError: Input and output paths are required (unless using --info)")
        return 1
    
    # Convert directory
    if args.directory:
        if not os.path.isdir(args.input):
            print(f"Error: Input path is not a directory: {args.input}")
            return 1
        
        print(f"\nConverting OneNote files from: {args.input}")
        print(f"Output directory: {args.output}")
        print()
        
        count = convert_onenote_directory(
            args.input,
            args.output,
            overwrite=not args.no_overwrite,
            verbose=args.verbose
        )
        
        print(f"\n✓ Successfully converted {count} file(s)")
        return 0 if count > 0 else 1
    
    # Convert single file
    else:
        if not os.path.isfile(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        if not args.input.lower().endswith('.one'):
            print(f"Warning: Input file does not have .one extension: {args.input}")
        
        print(f"\nConverting: {args.input}")
        print(f"Output: {args.output}")
        print()
        
        success = convert_onenote_to_pdf(
            args.input,
            args.output,
            verbose=args.verbose
        )
        
        if success:
            print(f"\n✓ Successfully converted to: {args.output}")
            return 0
        else:
            print("\n✗ Conversion failed")
            return 1


if __name__ == '__main__':
    sys.exit(main())
