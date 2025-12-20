# Platonic Ideal

A proof-of-concept for extracting token vectors from language model embedding matrices without downloading the full model.

## Purpose

This project enables efficient extraction of **non-contextual token embeddings** from large language models (currently tested with Qwen/Qwen2.5-0.5B). Unlike contextual embeddings (which require a full forward pass through the model), token vectors are the raw, static representations stored in the model's embedding matrix—the "platonic ideal" of how each token is represented before any contextual processing.

### What are Token Vectors?

Token vectors are the initial embeddings assigned to each token in the vocabulary. They represent:
- The **base representation** of each token before contextualization
- A mapping from token IDs to dense vector representations
- The first layer of the transformer model: `E[token_id] → vector`

These vectors are useful for:
- Analyzing token similarity and relationships
- Understanding the model's vocabulary structure
- Building lightweight token-based features without running the full model
- Research into embedding space geometry

## How It Works

Rather than loading entire multi-GB models, this project:
1. **Detects** whether the model uses sharded or single-file safetensors format
2. **Downloads** only the embedding matrix (not the full model weights)
3. **Loads** embeddings efficiently on CPU using memory-mapped files
4. **Extracts** token vectors for text or token IDs
5. **Converts** special formats (like bfloat16) to standard float32 numpy arrays

For Qwen/Qwen2.5-0.5B:
- Full model: ~1GB
- Just embeddings: Same file (single safetensors), but loaded efficiently
- Vocabulary size: 151,936 tokens
- Embedding dimension: 896

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

### Extract embeddings for specific token IDs

```bash
python extract_embeddings.py --token-id 100
```

Output:
```
Token ID: 100
Vector shape: (896,)
Vector (first 20 dims): [-0.01177979  0.03857422  0.03979492 ...]
Vector statistics:
  Mean: -0.000764
  Std: 0.016586
  Min: -0.049072
  Max: 0.048828
```

### Extract embeddings for text

```bash
python extract_embeddings.py --text "Hello world"
```

Output:
```
Input text: Hello world
Token IDs: [9707, 1879]
Number of tokens: 2
Embedding shape: (2, 896)

Token breakdown:
  Token 0: 'Hello' (ID: 9707)
    Vector (first 10 dims): [-2.5146484e-02  5.5541992e-03 ...]
  Token 1: ' world' (ID: 1879)
    Vector (first 10 dims): [ 0.00830078 -0.00072479 ...]
```

### Use a different model

```bash
python extract_embeddings.py --repo-id "Qwen/Qwen2.5-1.5B" --text "Hello"
```

## What's Next

This is a minimal proof-of-concept. Future enhancements could include:
- Package structure for easy installation
- Vector similarity and clustering operations
- Support for more model architectures
- Efficient nearest-neighbor search with FAISS
- CLI improvements with better output formatting
- API for programmatic access

