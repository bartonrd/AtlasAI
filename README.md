# AtlasAI

A Retrieval-Augmented Generation (RAG) chatbot for querying technical documentation using local language models.

## Overview

AtlasAI is a C# application that integrates a Python-based LLM chatbot with a Streamlit UI. The chatbot uses local Hugging Face models to answer questions about technical documents stored in the `documents` folder.

## Features

- **Local LLM Processing**: Uses offline Hugging Face models (FLAN-T5) for text generation
- **RAG System**: Retrieves relevant context from PDF/DOCX documents before answering
- **Streamlit UI**: Interactive web-based chat interface
- **C# Wrapper**: Launches the chatbot through a .NET console application

## Prerequisites

### 1. .NET SDK
- .NET 10.0 or later
- Download from: https://dotnet.microsoft.com/download

### 2. Python
- Python 3.9 or later
- Ensure Python is in your system PATH

### 3. Python Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Hugging Face Models
Download the following models to your local machine:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Text Generation Model**: `google/flan-t5-base` (or `google/flan-t5-small` for faster CPU runs)

Update the model paths in `chatapp.py`:

```python
EMBEDDING_MODEL = r"C:\models\all-MiniLM-L6-v2"
LOCAL_TEXT_GEN_MODEL = r"C:\models\flan-t5-base"
```

You can download models using Python:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Download and save embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save(r'C:\models\all-MiniLM-L6-v2')

# Download and save text generation model
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
tokenizer.save_pretrained(r'C:\models\flan-t5-base')
model.save_pretrained(r'C:\models\flan-t5-base')
```

## Project Structure

```
AtlasAI/
├── AtlasAI/                 # C# console application
│   ├── Program.cs          # Main C# entry point
│   └── AtlasAI.csproj      # C# project file
├── documents/              # PDF/DOCX files for RAG
│   ├── distribution_model_manager_user_guide.pdf
│   └── adms-16-20-0-modeling-overview-and-converter-user-guide.pdf
├── chatapp.py              # Python Streamlit chatbot
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Building and Running

### Option 1: Run through C# Application (Recommended)

1. Build the C# application:
   ```bash
   cd AtlasAI
   dotnet build
   ```

2. Run the application:
   ```bash
   dotnet run
   ```

The C# application will automatically launch the Streamlit chatbot and open it in your default web browser.

### Option 2: Run Python Script Directly

```bash
streamlit run chatapp.py
```

## Usage

1. Once the Streamlit UI opens in your browser, you'll see a chat interface
2. Optionally upload additional PDF or DOCX files using the file uploader
3. Type your question in the chat input at the bottom
4. The chatbot will retrieve relevant context from the documents and generate an answer
5. View source citations in the expandable "Sources" section

## Configuration

Edit `chatapp.py` to customize:

- **TOP_K**: Number of document chunks to retrieve (default: 4)
- **CHUNK_SIZE**: Size of text chunks for splitting (default: 1000)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 150)
- **Model paths**: Update to point to your local Hugging Face models

## Adding Documents

Place PDF or DOCX files in the `documents/` folder. The chatbot will automatically load them when you ask questions.

## Troubleshooting

### "streamlit: command not found"
- Ensure Streamlit is installed: `pip install streamlit`
- Verify Python's Scripts folder is in your PATH

### "Failed to load local HF model"
- Verify model paths are correct in `chatapp.py`
- Ensure models are downloaded to the specified locations
- Check that the models are compatible (FLAN-T5 for text generation, sentence-transformers for embeddings)

### "No documents loaded"
- Verify PDF files exist in the `documents/` folder
- Check that file paths are correct
- Ensure PDFs contain extractable text (not scanned images without OCR)

## License

[Add your license here]

## Contact

[Add contact information here]