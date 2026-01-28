#!/usr/bin/env python3
"""
Environment setup script for AtlasAI.
Installs required Python packages and sets up Ollama.
"""

import subprocess
import sys
import os
import platform
import shutil


def run_command(cmd, check=True, shell=False):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install Python requirements."""
    print("\nInstalling Python requirements...")
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print(f"❌ requirements.txt not found at {requirements_file}")
        return False
    
    success, output = run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    
    if success:
        print("✓ Python requirements installed successfully")
        return True
    else:
        print(f"❌ Failed to install requirements:\n{output}")
        return False


def check_ollama():
    """Check if Ollama is installed."""
    print("\nChecking Ollama installation...")
    
    # Check if ollama command is available
    ollama_path = shutil.which("ollama")
    if ollama_path:
        print(f"✓ Ollama found at: {ollama_path}")
        return True
    
    print("⚠ Ollama not found in PATH")
    print("\nPlease install Ollama:")
    
    system = platform.system()
    if system == "Windows":
        print("  Download from: https://ollama.ai/download/windows")
    elif system == "Darwin":  # macOS
        print("  Run: brew install ollama")
        print("  Or download from: https://ollama.ai/download/mac")
    elif system == "Linux":
        print("  Run: curl -fsSL https://ollama.ai/install.sh | sh")
    else:
        print("  Visit: https://ollama.ai/download")
    
    return False


def check_ollama_running():
    """Check if Ollama service is running."""
    print("\nChecking if Ollama service is running...")
    
    # Try to connect to Ollama API
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama service is running")
            return True
    except Exception:
        pass
    
    print("⚠ Ollama service not running")
    print("\nTo start Ollama:")
    
    system = platform.system()
    if system == "Windows":
        print("  The Ollama service should start automatically after installation")
        print("  Or run: ollama serve")
    else:
        print("  Run: ollama serve")
    
    return False


def pull_default_model():
    """Pull the default model if Ollama is available."""
    print("\nChecking for default model (llama3.1:8b)...")
    
    # Check if model is already available
    success, output = run_command(["ollama", "list"])
    if success and "llama3.1:8b" in output:
        print("✓ Model llama3.1:8b already available")
        return True
    
    print("Pulling llama3.1:8b model (this may take a while)...")
    success, output = run_command(["ollama", "pull", "llama3.1:8b"])
    
    if success:
        print("✓ Model llama3.1:8b downloaded successfully")
        return True
    else:
        print(f"⚠ Failed to pull model:\n{output}")
        print("You can manually pull it later with: ollama pull llama3.1:8b")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("AtlasAI Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check Ollama
    ollama_installed = check_ollama()
    if ollama_installed:
        ollama_running = check_ollama_running()
        if ollama_running:
            # Try to pull default model
            pull_default_model()
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)
    print(f"✓ Python requirements installed")
    print(f"{'✓' if ollama_installed else '⚠'} Ollama {'installed' if ollama_installed else 'NOT installed'}")
    
    if not ollama_installed:
        print("\n⚠ IMPORTANT: Install Ollama to use AtlasAI")
    
    print("\nSetup complete! You can now run AtlasAI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
