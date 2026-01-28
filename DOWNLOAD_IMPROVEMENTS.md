# Download Utility Improvements

## Problem
Users in offline environments or with restricted internet access were getting unhelpful error messages when trying to download models with the `download_models.py` script.

## Solution
Enhanced the download utility with better error handling, connectivity checks, and comprehensive offline download instructions.

## New Features

### 1. Internet Connectivity Check
The script now checks for internet connectivity to Hugging Face before attempting downloads:

```bash
python download_models.py --preset utility
```

**Output when offline:**
```
Checking internet connectivity to Hugging Face...

======================================================================
⚠ WARNING: Cannot connect to huggingface.co
======================================================================

This script requires internet access to download models.
If you're in an offline environment, you have two options:

1. Download models on a machine with internet access and transfer them
2. Use the Hugging Face Hub CLI or manual download

See MODELS.md for detailed offline download instructions.

To check if models are already downloaded, use:
  python download_models.py --preset utility --check
======================================================================
```

### 2. Model Existence Check
New `--check` flag to verify if models are already downloaded locally:

```bash
python download_models.py --preset utility --check
```

**Output:**
```
Checking models for preset: utility
======================================================================

Embedding Model (sentence-transformers/all-mpnet-base-v2):
  ✓ Found at: C:\models\all-mpnet-base-v2

Text Generation Model (microsoft/Phi-3-mini-4k-instruct):
  ✓ Found at: C:\models\phi-3-mini

======================================================================

✓ All models are available locally!

To use these models, set environment variables:
  ATLASAI_EMBEDDING_MODEL=C:\models\all-mpnet-base-v2
  ATLASAI_TEXT_GEN_MODEL=C:\models\phi-3-mini
```

### 3. Skip Already Downloaded Models
The script automatically detects if a model is already downloaded and skips re-downloading:

```bash
python download_models.py --preset utility
```

**Output if model exists:**
```
======================================================================
Downloading embedding model: sentence-transformers/all-mpnet-base-v2
✓ Model already exists at: C:\models\all-mpnet-base-v2
  Skipping download for: sentence-transformers/all-mpnet-base-v2
```

### 4. Detailed Offline Instructions
When a connectivity error is detected, the script provides detailed instructions for manual download:

```
======================================================================
OFFLINE DOWNLOAD INSTRUCTIONS
======================================================================

Since you don't have internet connectivity, you can manually download
the model from another machine with internet access and transfer it.

On a machine with internet access:
----------------------------------------------------------------------

Python code to download:
```python
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Detect model type
config = AutoConfig.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')

# For causal LM models (Mistral, Phi, Llama):
# model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)

tokenizer.save_pretrained(r'C:\models\phi-3-mini')
model.save_pretrained(r'C:\models\phi-3-mini')
```

Alternatively, use Hugging Face Hub:
----------------------------------------------------------------------
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir C:\models\phi-3-mini

Or download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/main

Then transfer the entire model directory to:
  C:\models\phi-3-mini

See MODELS.md for more detailed instructions.
======================================================================
```

## Updated Documentation

### MODELS.md
Added comprehensive "Downloading Models" section with:
- Online download instructions using the utility script
- Three methods for offline/manual download:
  1. Download on another machine with Python
  2. Using Hugging Face CLI
  3. Manual download from website
- Step-by-step transfer instructions
- Troubleshooting guide

## Usage Examples

### Check Connectivity and Download
```bash
# The script will check connectivity automatically
python download_models.py --preset utility
```

### Skip Connectivity Check (Advanced)
```bash
# For environments where the check may give false negatives
python download_models.py --preset utility --skip-connectivity-check
```

### Verify Models Before Running Application
```bash
# Check if required models exist
python download_models.py --preset utility --check

# If successful, start the application
python -m atlasai_runtime
```

### List Available Presets
```bash
python download_models.py --list-presets
```

## Benefits

1. **Better User Experience**: Clear error messages with actionable solutions
2. **Offline Support**: Comprehensive instructions for manual downloads
3. **Efficiency**: Skip re-downloading models that already exist
4. **Troubleshooting**: Easy verification of model availability
5. **Documentation**: Complete guide for offline environments

## For Offline Environments

If you're working in an environment without internet access:

1. On a machine **with** internet:
   - Run `python download_models.py --preset utility`
   - Models will be saved to `C:\models\` (Windows) or `~/models/` (Linux/Mac)

2. Transfer the entire `models` directory to your offline machine

3. On the offline machine:
   - Verify: `python download_models.py --preset utility --check`
   - Configure: Set `ATLASAI_EMBEDDING_MODEL` and `ATLASAI_TEXT_GEN_MODEL`
   - Run: `python -m atlasai_runtime`

See `MODELS.md` for complete instructions and alternative methods.
