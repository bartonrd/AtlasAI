# AtlasAI Examples

This document provides examples of using AtlasAI's features.

## Basic RAG Query

```python
import requests

# Send a question to AtlasAI
response = requests.post("http://localhost:8000/chat", json={
    "message": "What is the Distribution Model Manager?"
})

result = response.json()
print(f"Intent: {result['intent']} (confidence: {result['intent_confidence']:.2f})")
print(f"Answer:\n{result['answer']}")
print(f"\nSources: {len(result['sources'])} documents")
```

## Task Execution Examples

### Get System Information

```python
import requests

response = requests.get("http://localhost:8000/task/system-info")
info = response.json()

print(f"OS: {info['system']}")
print(f"Python: {info['python_version']}")
print(f"Working Directory: {info['cwd']}")
```

### Execute a Command

```python
import requests

# Check Python version
response = requests.post("http://localhost:8000/task/execute", json={
    "command": "python --version",
    "timeout": 10
})

result = response.json()
if result['success']:
    print(f"Output: {result['stdout']}")
else:
    print(f"Error: {result['error']}")
```

### List Files in Directory

```python
import requests

response = requests.post("http://localhost:8000/task/execute", json={
    "command": "ls -la",
    "working_dir": "/path/to/directory",
    "timeout": 10
})

result = response.json()
print(result['stdout'])
```

## Using the Agent Workflow

```python
from atlasai_runtime.agent import create_agent_workflow
from atlasai_runtime.rag_engine import RAGEngine
from atlasai_runtime.task_executor import LocalTaskExecutor

# Initialize components
rag_engine = RAGEngine(
    documents_dir="./documents",
    onenote_runbook_path="./onenote_files",
    ollama_model="llama3.1:8b"
)

task_executor = LocalTaskExecutor()

# Create agent
agent = create_agent_workflow(rag_engine, task_executor)

# Process queries with automatic routing
result = agent.process_query("How do I check the Python version?")
print(result['final_answer'])
```

## Custom Ollama Model

```python
from atlasai_runtime.rag_engine import RAGEngine

# Use Qwen model for better technical content
engine = RAGEngine(
    documents_dir="./documents",
    onenote_runbook_path="./onenote_files",
    ollama_model="qwen2.5:7b",  # Different model
    embedding_model="mxbai-embed-large",
    top_k=10  # Retrieve more chunks
)

result = engine.query("Explain the grid topology")
print(result['answer'])
```

## Batch Document Processing

```python
from atlasai_runtime.rag_engine import RAGEngine
import os

engine = RAGEngine(
    documents_dir="./documents",
    onenote_runbook_path="./onenote_files"
)

# Query with additional documents
questions = [
    "What is SCADA?",
    "How to configure the database?",
    "Explain error code E001"
]

for question in questions:
    result = engine.query(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Intent: {result['intent']}")
    print("-" * 50)
```

## Programmatic OneNote Conversion

```python
from atlasai_runtime.onenote_converter import convert_onenote_directory

# Convert all OneNote files in a directory
count = convert_onenote_directory(
    source_dir="/path/to/onenote/files",
    output_dir="./documents/converted",
    overwrite=True,
    use_local_copies=True,
    local_copy_dir="./temp_copies"
)

print(f"Converted {count} OneNote files to PDF")
```

## Environment Configuration

### Using Different Models

```bash
# Use Mistral 7B instead of Llama
export ATLASAI_OLLAMA_MODEL="mistral:7b"

# Use smaller model for faster responses
export ATLASAI_OLLAMA_MODEL="llama3.1:3b"

# Use larger model for better quality (needs GPU)
export ATLASAI_OLLAMA_MODEL="llama3.1:13b"
```

### Custom Vector Store Location

```bash
# Store ChromaDB data in custom location
export ATLASAI_CHROMA_PERSIST_DIR="/data/atlasai/vectors"

# Use more chunks for retrieval
export ATLASAI_TOP_K="10"

# Adjust chunking parameters
export ATLASAI_CHUNK_SIZE="1000"
export ATLASAI_CHUNK_OVERLAP="200"
```

## C# Integration Example

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

public class AtlasAIClient
{
    private readonly HttpClient _client;
    
    public AtlasAIClient(string baseUrl = "http://localhost:8000")
    {
        _client = new HttpClient { BaseAddress = new Uri(baseUrl) };
    }
    
    public async Task<ChatResponse> SendQueryAsync(string message)
    {
        var request = new { message };
        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        var response = await _client.PostAsync("/chat", content);
        response.EnsureSuccessStatusCode();
        
        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<ChatResponse>(responseJson);
    }
    
    public async Task<TaskResponse> ExecuteTaskAsync(string command)
    {
        var request = new { command, timeout = 30 };
        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        var response = await _client.PostAsync("/task/execute", content);
        response.EnsureSuccessStatusCode();
        
        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<TaskResponse>(responseJson);
    }
}

public class ChatResponse
{
    public string answer { get; set; }
    public List<Source> sources { get; set; }
    public string intent { get; set; }
    public double intent_confidence { get; set; }
}

public class Source
{
    public int index { get; set; }
    public string source { get; set; }
    public string page { get; set; }
}

public class TaskResponse
{
    public bool success { get; set; }
    public string stdout { get; set; }
    public string stderr { get; set; }
    public int return_code { get; set; }
    public string error { get; set; }
}

// Usage
var client = new AtlasAIClient();
var response = await client.SendQueryAsync("What is ADMS?");
Console.WriteLine($"Answer: {response.answer}");
```

## Advanced: Custom Agent Workflow

```python
from atlasai_runtime.agent import SimpleAgent, AgentState
from typing import Dict, Any

class CustomAgent(SimpleAgent):
    """Extended agent with custom workflows."""
    
    def process_with_validation(self, query: str) -> Dict[str, Any]:
        """Process query with additional validation steps."""
        
        # Step 1: Initial processing
        state = self.process_query(query)
        
        # Step 2: Validate results
        if state['doc_results']:
            confidence = state['doc_results'].get('intent_confidence', 0)
            if confidence < 0.5:
                # Low confidence, try alternative approach
                state['final_answer'] = self._fallback_response(query)
        
        # Step 3: Add metadata
        state['metadata'] = {
            'processing_time': '1.2s',
            'model': 'llama3.1:8b',
            'chunks_retrieved': 6
        }
        
        return state
    
    def _fallback_response(self, query: str) -> str:
        """Provide fallback response for low-confidence queries."""
        return f"I'm not very confident about this answer. You asked: '{query}'"

# Usage
agent = CustomAgent(rag_engine, task_executor)
result = agent.process_with_validation("Complex technical query")
```

## Streamlit UI Customization

```python
import streamlit as st
import requests

st.title("Custom AtlasAI Interface")

# Custom sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"])
    top_k = st.slider("Chunks to retrieve", 1, 10, 6)

# Chat interface
query = st.text_input("Ask a question:")
if st.button("Send"):
    response = requests.post("http://localhost:8000/chat", json={"message": query})
    result = response.json()
    
    # Display intent
    st.info(f"Intent: {result['intent']} ({result['intent_confidence']:.2%})")
    
    # Display answer
    st.markdown(result['answer'])
    
    # Display sources
    if result['sources']:
        st.subheader("Sources")
        for src in result['sources']:
            st.text(f"{src['index']}. {src['source']} (page {src['page']})")
```

## Performance Optimization

### Caching Vector Store

```python
# ChromaDB automatically persists data
# No need to rebuild index on every run
engine = RAGEngine(
    documents_dir="./documents",
    chroma_persist_dir="./chroma_db"  # Reuses existing index
)
```

### Batch Processing with Ollama

```python
import ollama

# Pre-warm the model
ollama.generate(model="llama3.1:8b", prompt="test", keep_alive="5m")

# Process multiple queries
queries = ["Query 1", "Query 2", "Query 3"]
for query in queries:
    result = engine.query(query)
    # Model stays in memory between queries
```

### Parallel Task Execution

```python
import concurrent.futures
from atlasai_runtime.task_executor import LocalTaskExecutor

executor = LocalTaskExecutor()
commands = ["python --version", "git --version", "node --version"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results = pool.map(executor.execute_command, commands)
    for result in results:
        print(result['stdout'])
```
