# Platonic Ideal

A project for extracting token vectors from the DeepSeek-V3.2 model's embedding matrix.

## Purpose

This project enables efficient extraction of **non-contextual token embeddings** from the DeepSeek-V3.2 language model. Unlike contextual embeddings (which require a full forward pass through the model), token vectors are the raw, static representations stored in the model's embedding matrix—the "platonic ideal" of how each token is represented before any contextual processing.

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

### Why DeepSeek-V3.2?

DeepSeek-V3.2 is a state-of-the-art language model with a large vocabulary and high-dimensional embeddings. Extracting its token vectors provides insights into how modern language models represent tokens at their most fundamental level.

## Approach

Rather than loading the entire 690GB model, this project focuses on **efficiently loading only the embedding matrix** from the sharded model files. This allows for:
- Minimal disk space usage
- Fast access to token vectors
- No need for GPU resources (embeddings can be loaded on CPU)

## Status

This project is in development. The goal is to provide a clean interface for:
1. Downloading only the necessary model shard containing embeddings
2. Loading the embedding matrix efficiently
3. Extracting token vectors for any given text or token IDs
4. Performing basic operations on token vectors (similarity, clustering, etc.)

