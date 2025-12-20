#!/usr/bin/env python3
"""
Extract token embeddings from Qwen/Qwen2.5-0.5B model.
Downloads only the embedding matrix shard, not the full model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer


def find_embedding_shard(repo_id: str = "Qwen/Qwen2.5-0.5B") -> Tuple[str, str]:
    """
    Download and parse the model index to find which shard contains the embedding tensor.
    If the model is not sharded, returns the single model file.
    
    Returns:
        Tuple of (tensor_name, shard_filename)
    """
    print(f"📥 Checking model structure for {repo_id}...")
    
    # Try to download the index file (for sharded models)
    try:
        index_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            repo_type="model"
        )
        
        # Parse the index
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        
        # Look for embedding tensor (common names)
        embedding_candidates = [
            "model.embed_tokens.weight",
            "model.embeddings.word_embeddings.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
        ]
        
        for tensor_name in embedding_candidates:
            if tensor_name in weight_map:
                shard_name = weight_map[tensor_name]
                print(f"✅ Found embedding tensor: {tensor_name}")
                print(f"   Located in shard: {shard_name}")
                return tensor_name, shard_name
        
        # If not found, list all tensors with "embed" in the name
        print("⚠️  Standard embedding tensor names not found. Searching for candidates...")
        candidates = {k: v for k, v in weight_map.items() if "embed" in k.lower()}
        
        if candidates:
            print(f"Found {len(candidates)} tensor(s) with 'embed' in name:")
            for tensor_name, shard_name in candidates.items():
                print(f"  - {tensor_name} -> {shard_name}")
            # Use the first one
            tensor_name, shard_name = list(candidates.items())[0]
            print(f"🔧 Using: {tensor_name}")
            return tensor_name, shard_name
        
        raise ValueError("Could not find embedding tensor in model. Check the model architecture.")
    
    except Exception as e:
        if "404" in str(e) or "EntryNotFoundError" in str(e):
            # Model is not sharded, try single file
            print(f"ℹ️  No index file found - checking for single-file model...")
            
            # For single-file models, we need to inspect the file to find tensor names
            # We'll use a common default and let the load function discover it
            embedding_candidates = [
                "model.embed_tokens.weight",
                "model.embeddings.word_embeddings.weight",
                "transformer.wte.weight",
                "embeddings.word_embeddings.weight",
            ]
            
            # Return the first candidate and the single model file
            print(f"✅ Single-file model detected: model.safetensors")
            print(f"   Will search for embedding tensor during load")
            return None, "model.safetensors"  # None means we'll detect it during load
        else:
            raise


def download_files(repo_id: str, shard_name: str) -> Tuple[Path, Path]:
    """
    Download the embedding shard and tokenizer files.
    
    Returns:
        Tuple of (shard_path, tokenizer_cache_dir)
    """
    print(f"\n📦 Downloading embedding shard: {shard_name}")
    
    # Download the shard containing embeddings
    shard_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_name,
        repo_type="model"
    )
    print(f"✅ Shard downloaded: {shard_path}")
    
    # Tokenizer will be downloaded automatically by AutoTokenizer
    # Just return the cache directory
    cache_dir = Path(shard_path).parent
    
    return Path(shard_path), cache_dir


def load_embedding_matrix(shard_path: Path, tensor_name: Optional[str]) -> np.ndarray:
    """
    Load the embedding matrix from the shard file.
    If tensor_name is None, will search for common embedding tensor names.
    
    Returns:
        numpy array of shape (vocab_size, embedding_dim)
    """
    print(f"\n🔧 Loading embedding matrix from shard...")
    
    # Use PyTorch framework to handle bfloat16 and other special dtypes
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        # If tensor_name is not specified, search for it
        if tensor_name is None:
            print("   Searching for embedding tensor...")
            all_tensors = f.keys()
            
            # Look for embedding tensor (common names)
            embedding_candidates = [
                "model.embed_tokens.weight",
                "model.embeddings.word_embeddings.weight",
                "transformer.wte.weight",
                "embeddings.word_embeddings.weight",
            ]
            
            for candidate in embedding_candidates:
                if candidate in all_tensors:
                    tensor_name = candidate
                    print(f"   ✅ Found: {tensor_name}")
                    break
            
            # If still not found, search for any tensor with "embed" in name
            if tensor_name is None:
                embed_tensors = [t for t in all_tensors if "embed" in t.lower()]
                if embed_tensors:
                    tensor_name = embed_tensors[0]
                    print(f"   ✅ Found: {tensor_name}")
                else:
                    raise ValueError(f"Could not find embedding tensor. Available tensors: {list(all_tensors)[:10]}")
        
        # Get the embedding tensor as PyTorch tensor
        embeddings_pt = f.get_tensor(tensor_name)
        
        # Convert to float32 for better numpy compatibility
        if embeddings_pt.dtype != torch.float32:
            print(f"   Converting from {embeddings_pt.dtype} to float32...")
            embeddings_pt = embeddings_pt.float()
        
        # Convert to numpy
        embeddings = embeddings_pt.numpy()
    
    print(f"✅ Loaded embeddings with shape: {embeddings.shape}")
    print(f"   Vocabulary size: {embeddings.shape[0]}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Data type: {embeddings.dtype}")
    
    return embeddings


def get_vector(token_id: int, embeddings: np.ndarray) -> np.ndarray:
    """
    Extract the embedding vector for a single token ID.
    """
    if token_id < 0 or token_id >= embeddings.shape[0]:
        raise ValueError(f"Token ID {token_id} out of range [0, {embeddings.shape[0]-1}]")
    
    return embeddings[token_id]


def text_to_vectors(text: str, tokenizer, embeddings: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Convert text to token embeddings.
    
    Returns:
        Tuple of (token_ids, embedding_vectors)
    """
    # Tokenize the text
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    
    # Get embeddings for each token
    vectors = embeddings[token_ids]
    
    return token_ids, vectors


def main():
    parser = argparse.ArgumentParser(
        description="Extract token embeddings from Qwen/Qwen2.5-0.5B"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to extract embeddings for"
    )
    parser.add_argument(
        "--token-id",
        type=int,
        help="Token ID to extract embedding for"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face model repository ID"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and args.token_id is None:
        parser.error("Must provide either --text or --token-id")
    
    try:
        # Step 1: Find which shard contains embeddings
        tensor_name, shard_name = find_embedding_shard(args.repo_id)
        
        # Step 2: Download the shard and tokenizer
        shard_path, cache_dir = download_files(args.repo_id, shard_name)
        
        # Step 3: Load embedding matrix
        embeddings = load_embedding_matrix(shard_path, tensor_name)
        
        # Step 4: Load tokenizer (if needed for text input)
        tokenizer = None
        if args.text:
            print(f"\n🔤 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
            print("✅ Tokenizer loaded")
        
        # Step 5: Extract vectors
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        if args.text:
            token_ids, vectors = text_to_vectors(args.text, tokenizer, embeddings)
            print(f"\nInput text: {args.text}")
            print(f"Token IDs: {token_ids}")
            print(f"Number of tokens: {len(token_ids)}")
            print(f"Embedding shape: {vectors.shape}")
            print(f"\nToken breakdown:")
            for i, token_id in enumerate(token_ids):
                token_text = tokenizer.decode([token_id])
                print(f"  Token {i}: '{token_text}' (ID: {token_id})")
                print(f"    Vector (first 10 dims): {vectors[i][:10]}")
        
        elif args.token_id is not None:
            vector = get_vector(args.token_id, embeddings)
            print(f"\nToken ID: {args.token_id}")
            print(f"Vector shape: {vector.shape}")
            print(f"Vector (first 20 dims): {vector[:20]}")
            print(f"Vector (last 10 dims): {vector[-10:]}")
            print(f"\nVector statistics:")
            print(f"  Mean: {vector.mean():.6f}")
            print(f"  Std: {vector.std():.6f}")
            print(f"  Min: {vector.min():.6f}")
            print(f"  Max: {vector.max():.6f}")
        
        print("\n" + "="*60)
        print("✅ SUCCESS: Token embeddings extracted!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

