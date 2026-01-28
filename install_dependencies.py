"""
Dependency installer for AtlasAI.
Checks and installs required Python packages and Ollama.
"""

import sys
import subprocess
import os
import platform
import urllib.request
import json


def check_python_version():
    """Check if Python version is >= 3.9"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"ERROR: Python 3.9+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} OK")
    return True


def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✓ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("ERROR: pip is not available")
        return False


def install_requirements():
    """Install Python packages from requirements.txt"""
    print("\n" + "="*70)
    print("Installing Python dependencies...")
    print("="*70)
    
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"ERROR: requirements.txt not found at {requirements_path}")
        return False
    
    try:
        # Use pip install with upgrade flag
        cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path, "--upgrade"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("ERROR: Failed to install Python packages")
            print(result.stderr)
            return False
        
        print("✓ Python packages installed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Exception during package installation: {e}")
        return False


def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n" + "="*70)
    print("Checking Ollama installation...")
    print("="*70)
    
    # Check if ollama command exists
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Ollama is installed: {result.stdout.strip()}")
            return check_ollama_service()
        else:
            print("⚠ Ollama command not found")
            return False
    except FileNotFoundError:
        print("⚠ Ollama is not installed")
        print_ollama_install_instructions()
        return False
    except subprocess.TimeoutExpired:
        print("⚠ Ollama command timed out")
        return False


def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        # Try to connect to Ollama API
        import urllib.request
        import urllib.error
        
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                print("✓ Ollama service is running")
                return True
    except urllib.error.URLError:
        print("⚠ Ollama service is not running")
        print("  Please start Ollama service:")
        print("  - Windows/Mac: Start Ollama application")
        print("  - Linux: Run 'ollama serve' in a separate terminal")
        return False
    except Exception as e:
        print(f"⚠ Could not check Ollama service: {e}")
        return False


def print_ollama_install_instructions():
    """Print instructions for installing Ollama"""
    system = platform.system().lower()
    
    print("\nOllama is required but not installed.")
    print("\nTo install Ollama:")
    
    if system == "windows":
        print("  1. Download from: https://ollama.com/download/windows")
        print("  2. Run the installer")
        print("  3. Restart your terminal")
    elif system == "darwin":  # macOS
        print("  1. Download from: https://ollama.com/download/mac")
        print("  2. Run the installer")
        print("  3. Ollama will start automatically")
    elif system == "linux":
        print("  1. Run: curl -fsSL https://ollama.com/install.sh | sh")
        print("  2. Start the service: ollama serve")
    else:
        print("  Visit: https://ollama.com/download")
    
    print("\nAfter installation, run this program again.")


def check_ollama_model():
    """Check if a supported model is available"""
    print("\n" + "="*70)
    print("Checking Ollama models...")
    print("="*70)
    
    supported_models = [
        "llama3.1:8b-instruct-q4_0",
        "llama3.1:8b",
        "qwen2.5:7b-instruct",
        "qwen2.5:7b",
        "mistral:7b-instruct",
        "mistral:7b"
    ]
    
    try:
        # Get list of installed models
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("⚠ Could not list Ollama models")
            return False
        
        installed_models = result.stdout.lower()
        
        # Check if any supported model is installed
        found_models = []
        for model in supported_models:
            model_base = model.split(':')[0]
            if model_base in installed_models:
                found_models.append(model)
        
        if found_models:
            print(f"✓ Found compatible model(s): {', '.join(found_models)}")
            return True
        else:
            print("⚠ No compatible models found")
            print("\nRecommended models (install one):")
            print("  ollama pull llama3.1:8b-instruct-q4_0  (Recommended, ~4.7GB)")
            print("  ollama pull qwen2.5:7b-instruct        (Alternative, ~4.7GB)")
            print("  ollama pull mistral:7b-instruct        (Alternative, ~4.1GB)")
            print("\nThe application will attempt to use an available model.")
            return True  # Don't fail - just warn
            
    except FileNotFoundError:
        print("⚠ Ollama not found")
        return False
    except subprocess.TimeoutExpired:
        print("⚠ Ollama command timed out")
        return False


def main():
    """Main installation routine"""
    print("="*70)
    print("AtlasAI Dependency Installer")
    print("="*70)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check pip
    if not check_pip():
        success = False
    
    # Install Python requirements
    if success:
        if not install_requirements():
            success = False
    
    # Check Ollama
    ollama_ok = check_ollama()
    if ollama_ok:
        check_ollama_model()
    else:
        print("\n⚠ WARNING: Ollama is not installed or not running")
        print("The application requires Ollama to function.")
        success = False
    
    print("\n" + "="*70)
    if success:
        print("✓ All dependencies are ready!")
        print("="*70)
        return 0
    else:
        print("⚠ Some dependencies need attention")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
