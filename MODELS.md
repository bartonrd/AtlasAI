# Model Options for Electric Utility Domain

## Overview

AtlasAI supports multiple offline language models for processing electric utility technical documentation. This document describes the available models and their applicability to power systems, transmission systems, and distribution systems.

## Current Model Architecture

The system uses two types of models:
1. **Embedding Model**: Converts text into vector representations for document retrieval
2. **Text Generation Model**: Generates answers based on retrieved context

## Recommended Models for Electric Utility Domain

### Text Generation Models

#### 1. FLAN-T5 (Default - Currently Implemented)
- **Model**: `google/flan-t5-base` or `google/flan-t5-small`
- **Size**: Base (250M params), Small (80M params)
- **Best For**: General technical documentation, good starting point
- **Hardware**: Runs on CPU, 4-8GB RAM
- **Pros**: 
  - Fast inference on CPU
  - Good instruction following
  - Well-tested and stable
- **Cons**: 
  - Limited context window (512 tokens)
  - Less sophisticated reasoning than newer models
- **Domain Applicability**: Good for basic Q&A on ADMS, SCADA, DMS documentation

#### 2. Mistral-7B (Recommended for Better Performance)
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Size**: 7 billion parameters
- **Best For**: Complex technical documentation, step-by-step instructions
- **Hardware**: 16GB+ RAM, GPU recommended but works on CPU
- **Pros**:
  - Excellent technical vocabulary understanding
  - Better reasoning for troubleshooting scenarios
  - Larger context window (8k tokens)
  - Strong with structured information
- **Cons**:
  - Slower inference on CPU
  - Requires more memory
- **Domain Applicability**: Excellent for ADMS/DMS troubleshooting, error resolution, complex system explanations
- **Setup**:
  ```bash
  # Download using Hugging Face
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
  model.save_pretrained(r"C:\models\mistral-7b-instruct")
  tokenizer.save_pretrained(r"C:\models\mistral-7b-instruct")
  ```

#### 3. Phi-3-Mini (Recommended for Efficiency)
- **Model**: `microsoft/Phi-3-mini-4k-instruct`
- **Size**: 3.8 billion parameters
- **Best For**: Resource-constrained environments, fast responses
- **Hardware**: 8GB RAM, CPU-friendly
- **Pros**:
  - Compact yet powerful
  - Fast inference on CPU
  - Good technical understanding
  - Efficient for deployment
- **Cons**:
  - Smaller context window (4k tokens)
  - May struggle with very complex multi-step reasoning
- **Domain Applicability**: Good for quick lookups, standard procedures, configuration guides
- **Setup**:
  ```bash
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
  model.save_pretrained(r"C:\models\phi-3-mini")
  tokenizer.save_pretrained(r"C:\models\phi-3-mini")
  ```

#### 4. Llama-3.2 (For Advanced Use Cases)
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Size**: 3 billion parameters (also available in 1B)
- **Best For**: General technical documentation with good reasoning
- **Hardware**: 8-16GB RAM, GPU recommended
- **Pros**:
  - Strong reasoning capabilities
  - Good instruction following
  - Efficient for its capability level
- **Cons**:
  - Requires Hugging Face authentication
  - May be slower than Phi-3
- **Domain Applicability**: Good all-around model for utility documentation
- **Setup**:
  ```bash
  # Requires Hugging Face token
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token="YOUR_HF_TOKEN")
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token="YOUR_HF_TOKEN")
  model.save_pretrained(r"C:\models\llama-3.2-3b")
  tokenizer.save_pretrained(r"C:\models\llama-3.2-3b")
  ```

### Embedding Models

#### 1. all-MiniLM-L6-v2 (Default - Currently Implemented)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Size**: 22M parameters
- **Best For**: General document retrieval
- **Pros**: Fast, efficient, good general performance
- **Cons**: Not domain-specific
- **Domain Applicability**: Adequate for most utility documentation

#### 2. all-mpnet-base-v2 (Recommended for Better Retrieval)
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Size**: 110M parameters
- **Best For**: Higher quality document retrieval
- **Pros**: 
  - Better semantic understanding
  - Improved retrieval accuracy
  - Still relatively fast
- **Cons**: Larger memory footprint
- **Domain Applicability**: Better for technical terminology and complex queries
- **Setup**:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
  model.save(r'C:\models\all-mpnet-base-v2')
  ```

## Model Selection Guide

### For Different Use Cases

| Use Case | Recommended Text Gen | Recommended Embedding | Rationale |
|----------|---------------------|----------------------|-----------|
| Quick deployment | FLAN-T5-small | all-MiniLM-L6-v2 | Fastest, minimal resources |
| Balanced performance | Phi-3-Mini | all-MiniLM-L6-v2 | Good quality, reasonable speed |
| Best quality | Mistral-7B | all-mpnet-base-v2 | Highest accuracy for complex queries |
| Resource constrained | FLAN-T5-base | all-MiniLM-L6-v2 | Works on older hardware |
| Error troubleshooting | Mistral-7B | all-mpnet-base-v2 | Better reasoning for diagnostics |

### Hardware Requirements

| Configuration | RAM | GPU | Speed | Quality |
|--------------|-----|-----|-------|---------|
| Minimal | 8GB | No | Fast | Basic |
| Recommended | 16GB | Optional | Medium | Good |
| Optimal | 32GB+ | Yes (8GB+) | Fast | Excellent |

## Domain-Specific Considerations

### Electric Utility Documentation Types

1. **ADMS/DMS User Guides**
   - Best model: Mistral-7B or Phi-3-Mini
   - Need good step-by-step instruction following

2. **Error Logs & Troubleshooting**
   - Best model: Mistral-7B
   - Requires strong reasoning and technical understanding

3. **Configuration Documentation**
   - Best model: Phi-3-Mini or FLAN-T5-base
   - Structured information, less reasoning required

4. **System Architecture & Concepts**
   - Best model: Mistral-7B or Llama-3.2
   - Needs comprehensive explanation capabilities

5. **SCADA/Telemetry Documentation**
   - Best model: Phi-3-Mini or Mistral-7B
   - Technical precision important

## Using Different Models

### Configuration via Environment Variables

```bash
# Windows (PowerShell)
$env:ATLASAI_TEXT_GEN_MODEL="C:\models\mistral-7b-instruct"
$env:ATLASAI_EMBEDDING_MODEL="C:\models\all-mpnet-base-v2"
python -m atlasai_runtime

# Unix/Linux/macOS
export ATLASAI_TEXT_GEN_MODEL="/home/user/models/mistral-7b-instruct"
export ATLASAI_EMBEDDING_MODEL="/home/user/models/all-mpnet-base-v2"
python -m atlasai_runtime
```

### Model Type Detection

The system automatically detects the model architecture:
- Seq2Seq models (FLAN-T5): Uses `AutoModelForSeq2SeqLM`
- Causal LM models (Mistral, Phi-3, Llama): Uses `AutoModelForCausalLM`

## Future Enhancements

### Domain-Specific Fine-Tuning

For organizations with extensive utility documentation, consider fine-tuning models on:
- Internal ADMS/DMS documentation
- Historical incident reports and resolutions
- Standard operating procedures
- Regulatory compliance documents

#### Fine-Tuning Resources:
- **Dataset Creation**: Use existing Q&A pairs from support tickets
- **Tools**: Hugging Face Trainer, LoRA for efficient fine-tuning
- **Time**: 2-8 hours depending on dataset size and hardware
- **Benefits**: 20-40% improvement in domain-specific accuracy

### Knowledge Graph Integration

Advanced deployments can enhance models with:
- Equipment topology graphs
- Fault attribution networks
- Asset relationship databases

This reduces hallucinations and improves accuracy for specific utility infrastructure.

## Benchmarking & Validation

### Testing New Models

When evaluating a new model, test with:

1. **Error Resolution Queries**
   - "How do I fix point mapping errors in Model Manager?"
   - "What causes SCADA communication failures?"

2. **Concept Explanation Queries**
   - "What is Distribution Model Manager?"
   - "Explain ADMS state estimation"

3. **How-To Queries**
   - "How do I import CIM files?"
   - "Steps to configure measurement points"

4. **Performance Metrics**
   - Response time
   - Answer quality (subjective)
   - Source relevance
   - Technical accuracy

## Resources

- **Research Papers**: 
  - "Large Language Models for Power System Applications" (arXiv 2024)
  - "LLM4DistReconfig" (NAACL 2025)
  
- **Benchmarks**:
  - ElecBench (Electric Power Domain LLM Benchmark)
  
- **Open Datasets**:
  - EPRI Transmission & Distribution AI Project datasets

## Support

For questions about model selection or performance issues, consider:
1. Hardware capabilities and constraints
2. Response time requirements
3. Query complexity in your documentation
4. Available setup time and resources

Start with FLAN-T5 (current default) and upgrade to Mistral-7B or Phi-3-Mini based on performance needs.
