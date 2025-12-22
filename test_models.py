#!/usr/bin/env python3
"""
Test word-group essence computation across multiple models to compare results.

This script helps evaluate whether larger models or different architectures
produce better semantic results for the word-group essence task.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import extract_embeddings
from transformers import AutoTokenizer
import sys

# Add word_group_essence_wordnet functions
sys.path.insert(0, str(Path(__file__).parent))
import word_group_essence_wordnet as wge


# Models to test (in order of download size)
DEFAULT_TEST_MODELS = [
    {
        "id": "qwen-0.5b",
        "repo_id": "Qwen/Qwen2.5-0.5B",
        "description": "Qwen 0.5B (baseline, already tested)",
    },
    {
        "id": "tinyllama-1.1b",
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B (Llama architecture)",
    },
    {
        "id": "qwen-1.5b",
        "repo_id": "Qwen/Qwen2.5-1.5B",
        "description": "Qwen 1.5B (larger embeddings)",
    },
    {
        "id": "qwen-3b",
        "repo_id": "Qwen/Qwen2.5-3B",
        "description": "Qwen 3B (sharded, bilingual)",
    },
    {
        "id": "phi-2",
        "repo_id": "microsoft/phi-2",
        "description": "Microsoft Phi-2 2.7B (reasoning-focused)",
    },
]


def test_model(
    model_config: Dict[str, str],
    test_input: Dict[str, Any],
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Run word-group essence computation on a single model.
    
    Returns:
        Dict with model info, top words, and timing
    """
    repo_id = model_config["repo_id"]
    model_id = model_config["id"]
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_id}")
    print(f"  Repo: {repo_id}")
    print(f"  Description: {model_config['description']}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        # Prepare input
        doc = {
            "repo_id": repo_id,
            "groups": test_input["groups"]
        }
        
        # Compute essence (loads model internally)
        print(f"🧮 Computing word-group essence (includes model loading)...")
        
        result = wge.compute_essence_wordnet(
            doc=doc,
            repo_id_override=None,
            top_k_words=top_k,
            wordnet_pos="n,v",
            exclude_input=True,
            exclude_substrings=True,
            min_word_chars=3,
            max_token_len=6,
        )
        
        total_time = time.time() - start_time
        
        print(f"✅ Computation complete: {total_time:.1f}s total")
        
        vocab_size = result["_output"]["model"]["vocab_size"]
        embedding_dim = result["_output"]["model"]["embedding_dim"]
        
        return {
            "model_id": model_id,
            "repo_id": repo_id,
            "description": model_config["description"],
            "status": "success",
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "total_time_s": round(total_time, 2),
            "top_words": result["_output"]["top_words"][:top_k],
            "overall_norm": result["_output"]["overall"]["norm"],
            "num_groups": len(result["_output"]["groups"]),
            "wordnet_stats": result["_output"]["wordnet_stats"],
        }
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "model_id": model_id,
            "repo_id": repo_id,
            "description": model_config["description"],
            "status": "failed",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test word-group essence computation across multiple models"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with word groups (no repo_id needed)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for comparison results (default: print to stdout)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top words to return per model"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Model IDs to test (default: all predefined models)"
    )
    
    args = parser.parse_args()
    
    # Load test input
    with open(args.input, 'r', encoding='utf-8') as f:
        test_input = json.load(f)
    
    if "repo_id" in test_input:
        print("⚠️  Note: repo_id in input will be overridden for each model")
    
    # Determine which models to test
    if args.models:
        # Filter to requested models
        models_to_test = [m for m in DEFAULT_TEST_MODELS if m["id"] in args.models]
        if len(models_to_test) == 0:
            print(f"❌ No matching models found. Available: {[m['id'] for m in DEFAULT_TEST_MODELS]}")
            sys.exit(1)
    else:
        models_to_test = DEFAULT_TEST_MODELS
    
    print(f"\n🚀 Testing {len(models_to_test)} model(s)")
    print(f"   Input groups: {len(test_input['groups'])}")
    print(f"   Top-k: {args.top_k}")
    
    # Run tests
    results = []
    for model_config in models_to_test:
        result = test_model(model_config, test_input, args.top_k)
        results.append(result)
        
        # Brief summary
        if result["status"] == "success":
            print(f"\n📊 Top 5 words for {result['model_id']}:")
            for i, w in enumerate(result["top_words"][:5], 1):
                print(f"   {i}. {w['word']:20s} sim={w['cosine_similarity']:.4f}")
    
    # Generate comparison report
    comparison = {
        "test_input": test_input,
        "test_params": {
            "top_k": args.top_k,
            "wordnet_pos": ["n", "v"],
            "exclude_input": True,
            "exclude_substrings": True,
        },
        "results": results,
        "summary": {
            "total_models": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
        }
    }
    
    # Output
    output_json = json.dumps(comparison, ensure_ascii=False, indent=2)
    
    if args.output:
        args.output.write_text(output_json, encoding='utf-8')
        print(f"\n✅ Results written to: {args.output}")
    else:
        print("\n" + "="*80)
        print("COMPARISON RESULTS (JSON)")
        print("="*80)
        print(output_json)
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Status':<10} {'Vocab':<10} {'Dim':<6} {'Time(s)':<10} {'Candidates':<12}")
    print("-"*80)
    
    for r in results:
        if r["status"] == "success":
            candidates = r.get("wordnet_stats", {}).get("candidates_kept", "N/A")
            print(f"{r['model_id']:<20} {r['status']:<10} {r['vocab_size']:<10,} {r['embedding_dim']:<6} {r['total_time_s']:<10.1f} {candidates:<12,}" if isinstance(candidates, int) else f"{r['model_id']:<20} {r['status']:<10} {r['vocab_size']:<10,} {r['embedding_dim']:<6} {r['total_time_s']:<10.1f} {candidates:<12}")
        else:
            print(f"{r['model_id']:<20} {r['status']:<10} {'N/A':<10} {'N/A':<6} {'N/A':<10} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("TOP WORDS COMPARISON (first 10 per model)")
    print("="*80)
    
    for r in results:
        if r["status"] == "success":
            print(f"\n{r['model_id']} ({r['embedding_dim']} dim):")
            for i, w in enumerate(r["top_words"][:10], 1):
                print(f"  {i:2d}. {w['word']:25s} sim={w['cosine_similarity']:.4f}")
    
    print("\n✅ Model comparison complete!")


if __name__ == "__main__":
    main()

