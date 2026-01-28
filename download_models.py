#!/usr/bin/env python3
"""
Model Download Utility for AtlasAI

This script helps download and save Hugging Face models locally for offline use.
Supports both embedding models and text generation models.
"""

import argparse
import os
import sys
from pathlib import Path


def download_embedding_model(model_name: str, save_path: str):
    """Download and save an embedding model."""
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
        print(f"✗ Error downloading embedding model: {e}")
        return False


def download_text_generation_model(model_name: str, save_path: str, model_type: str = "auto"):
    """Download and save a text generation model."""
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
        print(f"✗ Error downloading text generation model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models for AtlasAI offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default models
  python download_models.py --preset default
  
  # Download recommended models for electric utility domain
  python download_models.py --preset utility
  
  # Download specific models
  python download_models.py --embedding sentence-transformers/all-MiniLM-L6-v2 --save-embedding C:\\models\\all-MiniLM-L6-v2
  python download_models.py --text-gen mistralai/Mistral-7B-Instruct-v0.2 --save-text-gen C:\\models\\mistral-7b-instruct
  
  # List available presets
  python download_models.py --list-presets
        """
    )
    
    parser.add_argument("--preset", choices=["default", "utility", "minimal", "optimal"],
                       help="Use a preset configuration")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available preset configurations")
    parser.add_argument("--embedding", help="Embedding model name from Hugging Face")
    parser.add_argument("--save-embedding", help="Path to save embedding model")
    parser.add_argument("--text-gen", help="Text generation model name from Hugging Face")
    parser.add_argument("--save-text-gen", help="Path to save text generation model")
    parser.add_argument("--model-type", choices=["auto", "seq2seq", "causal"], default="auto",
                       help="Type of text generation model (default: auto-detect)")
    
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
    
    # Use preset or individual models
    if args.preset:
        if args.preset not in presets:
            print(f"Error: Unknown preset '{args.preset}'")
            return 1
        
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
