#!/bin/bash
# Start the AtlasAI Python runtime service
# Usage: ./start_runtime.sh [--host HOST] [--port PORT]

# Default values
HOST="${ATLASAI_RUNTIME_HOST:-127.0.0.1}"
PORT="${ATLASAI_RUNTIME_PORT:-8000}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--host HOST] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --host HOST    Host to bind to (default: 127.0.0.1)"
            echo "  --port PORT    Port to bind to (default: 8000)"
            echo ""
            echo "Environment variables:"
            echo "  ATLASAI_RUNTIME_HOST      Override default host"
            echo "  ATLASAI_RUNTIME_PORT      Override default port"
            echo "  ATLASAI_DOCUMENTS_DIR     Documents directory path"
            echo "  ATLASAI_EMBEDDING_MODEL   Embedding model path"
            echo "  ATLASAI_TEXT_GEN_MODEL    Text generation model path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting AtlasAI Python Runtime..."
echo "Host: $HOST"
echo "Port: $PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the Python runtime module
python -m atlasai_runtime --host "$HOST" --port "$PORT"
