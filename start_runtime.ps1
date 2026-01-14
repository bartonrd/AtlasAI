# Start the AtlasAI Python runtime service
# Usage: .\start_runtime.ps1 [-Host HOST] [-Port PORT]

param(
    [string]$Host = $env:ATLASAI_RUNTIME_HOST,
    [int]$Port = [int]$env:ATLASAI_RUNTIME_PORT,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "Usage: .\start_runtime.ps1 [-Host HOST] [-Port PORT]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Host HOST    Host to bind to (default: 127.0.0.1)"
    Write-Host "  -Port PORT    Port to bind to (default: 8000)"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "Environment variables:"
    Write-Host "  ATLASAI_RUNTIME_HOST      Override default host"
    Write-Host "  ATLASAI_RUNTIME_PORT      Override default port"
    Write-Host "  ATLASAI_DOCUMENTS_DIR     Documents directory path"
    Write-Host "  ATLASAI_EMBEDDING_MODEL   Embedding model path"
    Write-Host "  ATLASAI_TEXT_GEN_MODEL    Text generation model path"
    exit 0
}

# Set defaults if not provided
if (-not $Host) {
    $Host = "127.0.0.1"
}

if ($Port -eq 0) {
    $Port = 8000
}

Write-Host "Starting AtlasAI Python Runtime..."
Write-Host "Host: $Host"
Write-Host "Port: $Port"
Write-Host ""
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Run the Python runtime module
python -m atlasai_runtime --host $Host --port $Port
