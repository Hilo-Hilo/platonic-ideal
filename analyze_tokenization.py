#!/usr/bin/env python3
"""
Analyze how the tokenizer handles individual words.
Useful for studying tokenizer behavior on sensitive or edge-case content.
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer


def analyze_word(word: str, tokenizer, show_vectors: bool = False) -> dict:
    """
    Analyze how a single word is tokenized.
    
    Returns:
        Dict with tokenization details
    """
    # Tokenize with and without special tokens
    token_ids_with_special = tokenizer.encode(word, add_special_tokens=True)
    token_ids_no_special = tokenizer.encode(word, add_special_tokens=False)
    
    # Decode individual tokens to see subword breakdown
    tokens = [tokenizer.decode([tid]) for tid in token_ids_no_special]
    
    return {
        "word": word,
        "token_ids": token_ids_no_special,
        "tokens": tokens,
        "num_tokens": len(token_ids_no_special),
        "token_ids_with_special": token_ids_with_special,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer behavior on individual words"
    )
    parser.add_argument(
        "--words",
        type=str,
        nargs="+",
        help="Words to analyze (space-separated)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="File containing words (one per line)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face model repository ID"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (CSV format)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.words and not args.file:
        parser.error("Must provide either --words or --file")
    
    # Load words
    words = []
    if args.words:
        words.extend(args.words)
    if args.file:
        if not args.file.exists():
            print(f"Error: File {args.file} not found", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            # Strip whitespace and skip empty lines
            file_words = [line.strip() for line in f if line.strip()]
            words.extend(file_words)
    
    if not words:
        print("Error: No words to analyze", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading tokenizer from {args.repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
    print(f"✅ Tokenizer loaded\n")
    
    # Analyze each word
    results = []
    print("="*80)
    print("TOKENIZATION ANALYSIS")
    print("="*80)
    
    for word in words:
        result = analyze_word(word, tokenizer)
        results.append(result)
        
        print(f"\nWord: '{result['word']}'")
        print(f"  Token IDs: {result['token_ids']}")
        print(f"  Number of tokens: {result['num_tokens']}")
        print(f"  Token breakdown:")
        for i, (tid, token) in enumerate(zip(result['token_ids'], result['tokens'])):
            print(f"    [{i}] ID {tid:6d} → '{token}'")
    
    print("\n" + "="*80)
    
    # Summary statistics
    print("\nSUMMARY:")
    print(f"  Total words analyzed: {len(results)}")
    print(f"  Average tokens per word: {sum(r['num_tokens'] for r in results) / len(results):.2f}")
    print(f"  Max tokens for single word: {max(r['num_tokens'] for r in results)}")
    print(f"  Min tokens for single word: {min(r['num_tokens'] for r in results)}")
    
    # Find words with unusual tokenization
    multi_token = [r for r in results if r['num_tokens'] > 1]
    if multi_token:
        print(f"\n  Words split into multiple tokens ({len(multi_token)}):")
        for r in multi_token[:10]:  # Show first 10
            print(f"    '{r['word']}' → {r['num_tokens']} tokens: {r['tokens']}")
    
    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'num_tokens', 'token_ids', 'tokens'])
            for r in results:
                writer.writerow([
                    r['word'],
                    r['num_tokens'],
                    ' '.join(map(str, r['token_ids'])),
                    ' | '.join(r['tokens'])
                ])
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()


