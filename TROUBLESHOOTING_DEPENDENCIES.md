# Troubleshooting: ModuleNotFoundError for Dependencies

## Issue: ModuleNotFoundError: No module named 'chromadb' (or other modules)

### Symptoms
When running AtlasAI, you see an error like:
```
ModuleNotFoundError: No module named 'chromadb'
```

Or similar errors for other dependencies like `ollama`, `fastapi`, `langchain`, etc.

### Root Cause
Python dependencies listed in `requirements.txt` have not been installed yet.

### Solution

The application is designed to install dependencies automatically. Choose one of these methods:

#### Method 1: Run via C# Application (Recommended)
This is the easiest method as it handles setup automatically:

```bash
cd AtlasAI
dotnet run
```

The C# `PythonRuntimeManager` will:
1. Automatically run `setup_env.py` before starting the Python runtime
2. Install all dependencies from `requirements.txt`
3. Verify the environment
4. Start the Python runtime

#### Method 2: Manual Setup (If needed)
If you're running the Python runtime directly or the automatic setup fails:

```bash
# Run the setup script manually
python setup_env.py
```

This will:
- Check Python version (requires Python 3.9+)
- Install all requirements from `requirements.txt`
- Verify Ollama installation (warns if missing)

#### Method 3: Direct pip Installation
If both methods above fail, install dependencies directly:

```bash
pip install -r requirements.txt
```

### Verification

After installation, verify all dependencies are available:

```bash
python -c "import chromadb; print('chromadb:', chromadb.__version__)"
python -c "import ollama; print('ollama: OK')"
python -c "import fastapi; print('fastapi: OK')"
```

### Dependencies List

The application requires these key dependencies (from `requirements.txt`):

- **chromadb>=0.4.22** - Vector database for RAG
- **ollama>=0.1.0** - Ollama Python client
- **fastapi>=0.104.0** - Web framework
- **langchain>=0.1.0** - LLM framework
- **langchain-community>=0.0.20** - LangChain community components
- Plus document loaders, UI components, and utilities

### Common Issues

#### Issue: "pip not found" or "pip install fails"

**Solution:**
```bash
# Use Python module syntax
python -m pip install -r requirements.txt

# Or update pip first
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Issue: "Permission denied" on Linux/macOS

**Solution:**
```bash
# Install to user directory
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Issue: Setup runs but dependencies still missing

**Possible causes:**
1. Multiple Python installations - check which Python is being used
2. Virtual environment not activated
3. Different Python version being used by C# vs command line

**Solution:**
```bash
# Check Python path
which python    # Linux/macOS
where python    # Windows

# Check Python version
python --version

# Verify pip is for the correct Python
python -m pip --version
```

### Prevention

To avoid this issue in the future:

1. **Always run via C# application** (`dotnet run`) - it handles setup automatically
2. **Use virtual environments** to isolate dependencies
3. **Keep dependencies updated**: `pip install -U -r requirements.txt`

### Still Having Issues?

If you continue to experience dependency errors:

1. **Check the setup output** - The C# application displays setup_env.py output
2. **Look for error messages** - Setup failures are logged with ❌ markers
3. **Run setup manually** with verbose output:
   ```bash
   python setup_env.py 2>&1 | tee setup_log.txt
   ```
4. **Check requirements.txt** exists in the project root
5. **Verify Python version** is 3.9 or higher

### Technical Details

**Setup Flow:**
1. User runs: `cd AtlasAI && dotnet run`
2. C# `PythonRuntimeManager.Start()` is called
3. `RunSetupScript()` executes `setup_env.py`
4. `setup_env.py` runs: `pip install -r requirements.txt`
5. Dependencies are installed to Python's site-packages
6. Python runtime starts with all dependencies available

**Setup Script Location:** `setup_env.py` in project root  
**Requirements File:** `requirements.txt` in project root  
**C# Integration:** `AtlasAI/PythonRuntimeManager.cs` line 185-241
