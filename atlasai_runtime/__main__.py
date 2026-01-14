"""
Main entry point for AtlasAI Runtime.

Usage:
    python -m atlasai_runtime [--host HOST] [--port PORT]
"""

import argparse
import sys

import uvicorn


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AtlasAI Runtime - Python service for RAG-based chat completion")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting AtlasAI Runtime on {args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            "atlasai_runtime.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
