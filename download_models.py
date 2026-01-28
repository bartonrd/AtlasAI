#!/usr/bin/env python3
"""
Model Download Utility for AtlasAI

This script helps download and save Hugging Face models locally for offline use.
Supports both embedding models and text generation models.

Note: This script requires internet connectivity to download models from Hugging Face.
For offline environments, see the manual download instructions in MODELS.md.
"""

import argparse
import os
import sys
import urllib.request
import socket
from pathlib import Path


def check_internet_connectivity(host="huggingface.co", timeout=5):
    """
    Check if there is internet connectivity to Hugging Face.
    
    Args:
        host: Host to check connectivity to
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if connected, False otherwise
    """
    try:
        # Try to resolve the hostname
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        
        # Try to make a connection
        urllib.request.urlopen(f"https://{host}", timeout=timeout)
        return True
    except (socket.gaierror, urllib.error.URLError, socket.timeout, OSError):
        return False


def check_model_exists(model_path: str) -> bool:
    """
    Check if a model already exists at the given path.
    
    Args:
        model_path: Path to check for existing model
        
    Returns:
        bool: True if model exists and appears valid, False otherwise
    """
    if not os.path.exists(model_path):
        return False
    
    # Check for common model files
    has_config = os.path.exists(os.path.join(model_path, "config.json"))
    has_model = (
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
        os.path.exists(os.path.join(model_path, "model.safetensors")) or
        len([f for f in os.listdir(model_path) if f.startswith("pytorch_model")]) > 0
    )
    
    return has_config and has_model


def print_offline_instructions(model_name: str, save_path: str, is_embedding: bool = False):
    """Print instructions for manually downloading models when offline."""
    print("\n" + "="*70)
    print("OFFLINE DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nSince you don't have internet connectivity, you can manually download")
    print("the model from another machine with internet access and transfer it.")
    print("\nOn a machine with internet access:")
    print("-" * 70)
    
    if is_embedding:
        print("\nPython code to download:")
        print(f"```python")
        print(f"from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{model_name}')")
        print(f"model.save(r'{save_path}')")
        print(f"```")
    else:
        print("\nPython code to download:")
        print(f"```python")
        print(f"from transformers import AutoTokenizer, AutoConfig")
        print(f"from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM")
        print(f"")
        print(f"# Detect model type")
        print(f"config = AutoConfig.from_pretrained('{model_name}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{model_name}')")
        print(f"")
        print(f"# For seq2seq models (T5, BART):")
        print(f"# model = AutoModelForSeq2SeqLM.from_pretrained('{model_name}')")
        print(f"")
        print(f"# For causal LM models (Mistral, Phi, Llama):")
        print(f"# model = AutoModelForCausalLM.from_pretrained('{model_name}', trust_remote_code=True)")
        print(f"")
        print(f"tokenizer.save_pretrained(r'{save_path}')")
        print(f"model.save_pretrained(r'{save_path}')")
        print(f"```")
    
    print("\nAlternatively, use Hugging Face Hub:")
    print("-" * 70)
    print(f"huggingface-cli download {model_name} --local-dir {save_path}")
    print("\nOr download from: https://huggingface.co/{model_name}/tree/main")
    
    print("\nThen transfer the entire model directory to:")
    print(f"  {save_path}")
    print("\nSee MODELS.md for more detailed instructions.")
    print("="*70 + "\n")


def download_embedding_model(model_name: str, save_path: str):
    """Download and save an embedding model."""
    # Check if model already exists
    if check_model_exists(save_path):
        print(f"✓ Model already exists at: {save_path}")
        print(f"  Skipping download for: {model_name}")
        return True
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Downloading embedding model: {model_name}")
        print(f"This may take a few minutes depending on your internet connection...")
        
        model = SentenceTransformer(model_name)
        
        print(f"Saving model to: {save_path}")
        # Ensure parent directory exists
        parent_dir = os.path.dirname(save_path)
        if parent_dir:  # Only create if there is a directory component
            os.makedirs(parent_dir, exist_ok=True)
        model.save(save_path)
        
        print(f"✓ Successfully downloaded and saved: {model_name}")
        print(f"  Location: {save_path}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error downloading embedding model: {error_msg}")
        
        # Provide helpful error message based on the error type
        if "couldn't connect" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n⚠ CONNECTIVITY ISSUE DETECTED")
            print("  This script requires internet access to download models.")
            print_offline_instructions(model_name, save_path, is_embedding=True)
        
        return False


def download_text_generation_model(model_name: str, save_path: str, model_type: str = "auto"):
    """Download and save a text generation model."""
    # Check if model already exists
    if check_model_exists(save_path):
        print(f"✓ Model already exists at: {save_path}")
        print(f"  Skipping download for: {model_name}")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
        
        print(f"Downloading text generation model: {model_name}")
        print(f"This may take several minutes depending on model size and internet connection...")
        
        # Load config to detect model type if not specified
        if model_type == "auto":
            print("Detecting model type...")
            config = AutoConfig.from_pretrained(model_name)
            detected_type = config.model_type.lower()
            print(f"Detected model type: {detected_type}")
            
            seq2seq_models = ["t5", "bart", "pegasus", "mbart", "led", "bigbird_pegasus"]
            is_seq2seq = detected_type in seq2seq_models
            model_type = "seq2seq" if is_seq2seq else "causal"
        
        # Determine if trust_remote_code is needed
        needs_trust = model_name.lower().startswith("microsoft/phi")
        
        if needs_trust:
            print("\n" + "="*70)
            print("SECURITY WARNING:")
            print("This model requires trust_remote_code=True, which allows execution")
            print("of arbitrary code from the model repository. Only proceed if you")
            print("trust the model source.")
            print("="*70 + "\n")
        
        print(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=needs_trust
        )
        
        print(f"Loading model (this is the large download)...")
        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=needs_trust
            )
        
        print(f"Saving model to: {save_path}")
        # Ensure parent directory exists
        parent_dir = os.path.dirname(save_path)
        if parent_dir:  # Only create if there is a directory component
            os.makedirs(parent_dir, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"✓ Successfully downloaded and saved: {model_name}")
        print(f"  Location: {save_path}")
        print(f"  Type: {model_type}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error downloading text generation model: {error_msg}")
        
        # Provide helpful error message based on the error type
        if "couldn't connect" in error_msg.lower() or "connection" in error_msg.lower():
            print("\n⚠ CONNECTIVITY ISSUE DETECTED")
            print("  This script requires internet access to download models.")
            print_offline_instructions(model_name, save_path, is_embedding=False)
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models for AtlasAI offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available presets
  python download_models.py --list-presets
  
  # Check if models for a preset already exist locally
  python download_models.py --preset utility --check
  
  # Download default models
  python download_models.py --preset default
  
  # Download recommended models for electric utility domain
  python download_models.py --preset utility
  
  # Download specific models
  python download_models.py --embedding sentence-transformers/all-MiniLM-L6-v2 --save-embedding C:\\models\\all-MiniLM-L6-v2
  python download_models.py --text-gen mistralai/Mistral-7B-Instruct-v0.2 --save-text-gen C:\\models\\mistral-7b-instruct

Note: This script requires internet connectivity. For offline environments,
      see MODELS.md for manual download instructions.
        """
    )
    
    parser.add_argument("--preset", choices=["default", "utility", "minimal", "optimal"],
                       help="Use a preset configuration")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available preset configurations")
    parser.add_argument("--check", action="store_true",
                       help="Check if models from a preset already exist locally")
    parser.add_argument("--embedding", help="Embedding model name from Hugging Face")
    parser.add_argument("--save-embedding", help="Path to save embedding model")
    parser.add_argument("--text-gen", help="Text generation model name from Hugging Face")
    parser.add_argument("--save-text-gen", help="Path to save text generation model")
    parser.add_argument("--model-type", choices=["auto", "seq2seq", "causal"], default="auto",
                       help="Type of text generation model (default: auto-detect)")
    parser.add_argument("--skip-connectivity-check", action="store_true",
                       help="Skip internet connectivity check (not recommended)")
    
    args = parser.parse_args()
    
    # Detect platform and adjust default paths
    import platform
    is_windows = platform.system() == "Windows"
    base_path = "C:\\models" if is_windows else os.path.expanduser("~/models")
    
    # Define presets with platform-appropriate paths
    presets = {
        "default": {
            "embedding": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(base_path, "all-MiniLM-L6-v2")),
            "text_gen": ("google/flan-t5-base", os.path.join(base_path, "flan-t5-base")),
            "description": "Default models - fast, efficient, good starting point"
        },
        "minimal": {
            "embedding": ("sentence-transformers/all-MiniLM-L6-v2", os.path.join(base_path, "all-MiniLM-L6-v2")),
            "text_gen": ("google/flan-t5-small", os.path.join(base_path, "flan-t5-small")),
            "description": "Minimal models - fastest, lowest resource usage"
        },
        "utility": {
            "embedding": ("sentence-transformers/all-mpnet-base-v2", os.path.join(base_path, "all-mpnet-base-v2")),
            "text_gen": ("microsoft/Phi-3-mini-4k-instruct", os.path.join(base_path, "phi-3-mini")),
            "description": "Recommended for electric utility documentation - balanced performance"
        },
        "optimal": {
            "embedding": ("sentence-transformers/all-mpnet-base-v2", os.path.join(base_path, "all-mpnet-base-v2")),
            "text_gen": ("mistralai/Mistral-7B-Instruct-v0.2", os.path.join(base_path, "mistral-7b-instruct")),
            "description": "Optimal models - best quality for complex technical queries"
        }
    }
    
    # List presets
    if args.list_presets:
        print("\nAvailable Presets:\n")
        for name, config in presets.items():
            print(f"  {name}:")
            print(f"    {config['description']}")
            print(f"    Embedding: {config['embedding'][0]}")
            print(f"    Text Gen: {config['text_gen'][0]}")
            print()
        return 0
    
    # Check models
    if args.check:
        if not args.preset:
            print("Error: --check requires --preset to specify which models to check")
            return 1
        
        if args.preset not in presets:
            print(f"Error: Unknown preset '{args.preset}'")
            return 1
        
        config = presets[args.preset]
        embedding_model, embedding_path = config["embedding"]
        text_gen_model, text_gen_path = config["text_gen"]
        
        print(f"\nChecking models for preset: {args.preset}")
        print("=" * 70)
        
        embedding_exists = check_model_exists(embedding_path)
        text_gen_exists = check_model_exists(text_gen_path)
        
        print(f"\nEmbedding Model ({embedding_model}):")
        if embedding_exists:
            print(f"  ✓ Found at: {embedding_path}")
        else:
            print(f"  ✗ Not found at: {embedding_path}")
        
        print(f"\nText Generation Model ({text_gen_model}):")
        if text_gen_exists:
            print(f"  ✓ Found at: {text_gen_path}")
        else:
            print(f"  ✗ Not found at: {text_gen_path}")
        
        print("\n" + "=" * 70)
        
        if embedding_exists and text_gen_exists:
            print("\n✓ All models are available locally!")
            print("\nTo use these models, set environment variables:")
            print(f"  ATLASAI_EMBEDDING_MODEL={embedding_path}")
            print(f"  ATLASAI_TEXT_GEN_MODEL={text_gen_path}")
            return 0
        else:
            print("\n⚠ Some models are missing. Use without --check to download them.")
            print("  Or see MODELS.md for manual download instructions.")
            return 1
    
    # Use preset or individual models
    if args.preset:
        if args.preset not in presets:
            print(f"Error: Unknown preset '{args.preset}'")
            return 1
        
        # Check internet connectivity first (unless skipped)
        if not args.skip_connectivity_check:
            print("Checking internet connectivity to Hugging Face...")
            if not check_internet_connectivity():
                print("\n" + "="*70)
                print("⚠ WARNING: Cannot connect to huggingface.co")
                print("="*70)
                print("\nThis script requires internet access to download models.")
                print("If you're in an offline environment, you have two options:\n")
                print("1. Download models on a machine with internet access and transfer them")
                print("2. Use the Hugging Face Hub CLI or manual download\n")
                print("See MODELS.md for detailed offline download instructions.")
                print("\nTo check if models are already downloaded, use:")
                print(f"  python download_models.py --preset {args.preset} --check")
                print("="*70 + "\n")
                return 1
            print("✓ Connected to Hugging Face\n")
        
        config = presets[args.preset]
        print(f"\nUsing preset: {args.preset}")
        print(f"Description: {config['description']}\n")
        
        embedding_model, embedding_path = config["embedding"]
        text_gen_model, text_gen_path = config["text_gen"]
        
        success = True
        
        # Download embedding model
        print("=" * 70)
        if not download_embedding_model(embedding_model, embedding_path):
            success = False
        
        print("\n" + "=" * 70)
        # Download text generation model
        if not download_text_generation_model(text_gen_model, text_gen_path, args.model_type):
            success = False
        
        print("\n" + "=" * 70)
        if success:
            print("\n✓ All models downloaded successfully!")
            print("\nTo use these models, set environment variables:")
            print(f"  ATLASAI_EMBEDDING_MODEL={embedding_path}")
            print(f"  ATLASAI_TEXT_GEN_MODEL={text_gen_path}")
        else:
            print("\n✗ Some models failed to download. Check errors above.")
            return 1
        
    elif args.embedding or args.text_gen:
        success = True
        
        if args.embedding:
            if not args.save_embedding:
                print("Error: --save-embedding required when using --embedding")
                return 1
            if not download_embedding_model(args.embedding, args.save_embedding):
                success = False
        
        if args.text_gen:
            if not args.save_text_gen:
                print("Error: --save-text-gen required when using --text-gen")
                return 1
            if not download_text_generation_model(args.text_gen, args.save_text_gen, args.model_type):
                success = False
        
        if not success:
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
