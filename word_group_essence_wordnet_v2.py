#!/usr/bin/env python3
"""
V2: Improved word-group essence computation with three mathematical enhancements:
1. Spherical averaging (normalize-then-average at token and group levels)
2. Per-group scoring (prevent positive/negative cancellation)
3. All-but-the-Top anisotropy correction (remove mean + top PCs)

Based on improvements documented in improvements.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoTokenizer

import extract_embeddings


DEFAULT_REPO_ID = "Qwen/Qwen2.5-0.5B"

DEFAULT_TOP_K_WORDS = 20
DEFAULT_CANDIDATE_BATCH = 4096
DEFAULT_WEIGHT_CLIP = 32.0
DEFAULT_MAX_TOKEN_LEN = 6
DEFAULT_MIN_WORD_CHARS = 3
DEFAULT_EXCLUDE_INPUT = True
DEFAULT_EXCLUDE_COMPONENT_WORDS = True
DEFAULT_EXCLUDE_SUBSTRINGS = True

# V2 defaults
DEFAULT_USE_SPHERICAL_MEAN = True
DEFAULT_USE_PER_GROUP_SCORING = True
DEFAULT_ABT_ENABLED = True
DEFAULT_ABT_NUM_COMPONENTS = 5

# WordNet POS tags: n (noun), v (verb), a (adj), r (adv), s (adj satellite)
DEFAULT_WORDNET_POS = "n,v"


_WORDLIKE_ASCII_RE = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
_WORD_COMPONENT_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def _is_finite_number(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _clip_weight(weight: float, clip_abs: Optional[float]) -> Tuple[float, bool]:
    if clip_abs is None:
        return weight, False
    if clip_abs <= 0:
        return weight, False
    if abs(weight) <= clip_abs:
        return weight, False
    return math.copysign(clip_abs, weight), True


def _tokenize_no_special(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns original if norm is too small."""
    norm = float(np.linalg.norm(v))
    if norm <= 1e-10:
        return v.astype(np.float32, copy=False)
    return (v / norm).astype(np.float32, copy=False)


def _mean_token_vectors(embeddings: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    """Original Euclidean mean (for backwards compatibility checks)."""
    token_ids = list(token_ids)
    if len(token_ids) == 0:
        raise ValueError("Tokenized to zero tokens")
    vecs = embeddings[token_ids]  # (T, D)
    return vecs.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)


def _mean_token_vectors_spherical(embeddings: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    """
    Spherical mean: normalize each token vector, average, then normalize result.
    This averages directions on the unit sphere rather than raw vectors.
    """
    token_ids = list(token_ids)
    if len(token_ids) == 0:
        raise ValueError("Tokenized to zero tokens")
    
    vecs = embeddings[token_ids]  # (T, D)
    
    # Normalize each token vector
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    vecs_normalized = vecs / norms
    
    # Average normalized vectors
    mean_vec = vecs_normalized.mean(axis=0, dtype=np.float32)
    
    # Normalize result
    return _normalize_vector(mean_vec)


def _validate_input(doc: Dict[str, Any]) -> None:
    if not isinstance(doc, dict):
        raise ValueError("Input JSON must be an object")
    groups = doc.get("groups")
    if not isinstance(groups, list) or len(groups) == 0:
        raise ValueError("Input JSON must contain non-empty `groups` array")
    for i, g in enumerate(groups):
        if not isinstance(g, dict):
            raise ValueError(f"Group at index {i} must be an object")
        entries = g.get("entries")
        if not isinstance(entries, list) or len(entries) == 0:
            raise ValueError(f"Group at index {i} must contain non-empty `entries` array")


def _normalize_word(s: str) -> str:
    return s.strip().lower()


def _extract_component_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_COMPONENT_RE.finditer(text)]


def _build_exclude_word_set(
    groups: List[Dict[str, Any]],
    *,
    exclude_input: bool,
    exclude_component_words: bool,
) -> set[str]:
    exclude: set[str] = set()
    if not exclude_input:
        return exclude
    for g in groups:
        entries = g.get("entries", [])
        for e in entries:
            if not isinstance(e, str):
                continue
            w = _normalize_word(e)
            if _WORDLIKE_ASCII_RE.fullmatch(w):
                exclude.add(w)
            if exclude_component_words:
                for cw in _extract_component_words(e):
                    if _WORDLIKE_ASCII_RE.fullmatch(cw):
                        exclude.add(cw)
    return exclude


def _compute_essence_vector_v2(
    doc: Dict[str, Any],
    embeddings: np.ndarray,
    tokenizer,
    *,
    weight_clip_abs: Optional[float],
    use_spherical_mean: bool,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[float], List[str], bool]:
    """
    V2: Compute group mean vectors using spherical averaging.
    
    Returns:
        - group_results: list of group metadata
        - group_mean_vectors: list of normalized group mean vectors (for per-group scoring)
        - group_weights: list of weights (for per-group scoring)
        - warnings: list of warning messages
        - any_weight_clipped: whether any weights were clipped
    """
    groups_in: List[Dict[str, Any]] = doc["groups"]

    group_results: List[Dict[str, Any]] = []
    group_mean_vectors: List[np.ndarray] = []
    group_weights: List[float] = []
    warnings: List[str] = []
    any_weight_clipped = False

    for gi, group in enumerate(groups_in):
        entries = group.get("entries", [])
        weight_in = group.get("weight", 1.0)
        if not _is_finite_number(weight_in):
            warnings.append(f"Group[{gi}] has non-finite weight; defaulting to 1.0")
            weight_in = 1.0
        weight_in_f = float(weight_in)
        weight_used, clipped = _clip_weight(weight_in_f, weight_clip_abs)
        any_weight_clipped = any_weight_clipped or clipped

        entry_infos: List[Dict[str, Any]] = []
        entry_vecs: List[np.ndarray] = []

        for ei, entry in enumerate(entries):
            if not isinstance(entry, str):
                warnings.append(f"Group[{gi}].entries[{ei}] is not a string; skipping")
                continue
            token_ids = _tokenize_no_special(tokenizer, entry)
            if len(token_ids) == 0:
                warnings.append(f"Group[{gi}].entries[{ei}] tokenized to zero tokens; skipping")
                continue
            try:
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
            except Exception:
                tokens = []

            # Use spherical or Euclidean mean based on flag
            if use_spherical_mean:
                entry_vec = _mean_token_vectors_spherical(embeddings, token_ids)
            else:
                entry_vec = _mean_token_vectors(embeddings, token_ids)
            
            entry_vecs.append(entry_vec)
            entry_infos.append(
                {"text": entry, "token_ids": [int(t) for t in token_ids], "tokens": tokens}
            )

        if len(entry_vecs) == 0:
            raise ValueError(f"Group[{gi}] has zero valid entries after tokenization")

        # Compute group mean with optional spherical averaging
        if use_spherical_mean:
            # Normalize each entry vector, average, then normalize
            entry_vecs_array = np.stack(entry_vecs, axis=0)
            norms = np.linalg.norm(entry_vecs_array, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            entry_vecs_normalized = entry_vecs_array / norms
            group_mean = entry_vecs_normalized.mean(axis=0, dtype=np.float32)
            group_mean = _normalize_vector(group_mean)
        else:
            group_mean = (np.stack(entry_vecs, axis=0).mean(axis=0, dtype=np.float32)).astype(
                np.float32, copy=False
            )

        # Store unnormalized group mean and weight separately for per-group scoring
        group_mean_vectors.append(group_mean)
        group_weights.append(weight_used)

        # Compute weighted version for metadata
        group_weighted = (group_mean * np.float32(weight_used)).astype(np.float32, copy=False)

        group_results.append(
            {
                "index": gi,
                "name": group.get("name"),
                "weight_input": float(weight_in_f),
                "weight_used": float(weight_used),
                "num_entries": int(len(entry_vecs)),
                "entries": entry_infos,
                "group_mean_norm": float(np.linalg.norm(group_mean)),
                "group_weighted_norm": float(np.linalg.norm(group_weighted)),
            }
        )

    return group_results, group_mean_vectors, group_weights, warnings, any_weight_clipped


def _load_wordnet():
    """
    Import WordNet. If corpora are missing, raise an error with a helpful message.
    """
    try:
        import nltk  # noqa: F401
        from nltk.corpus import wordnet as wn
    except Exception as e:
        raise RuntimeError(
            "WordNet mode requires NLTK. Install it with `pip install nltk`."
        ) from e

    # Check corpus availability.
    try:
        _ = wn.synsets("space", pos="n")
    except LookupError as e:
        raise RuntimeError(
            "NLTK WordNet data is not installed.\n"
            "Run:\n"
            "  python -m nltk.downloader wordnet omw-1.4\n"
            "Then retry."
        ) from e

    return wn


def _parse_pos_list(pos_csv: str) -> List[str]:
    out: List[str] = []
    for p in (pos_csv or "").split(","):
        p = p.strip().lower()
        if not p:
            continue
        if p not in {"n", "v", "a", "r", "s"}:
            raise ValueError(f"Unsupported WordNet POS: {p}")
        out.append(p)
    if not out:
        out = ["n", "v"]
    # Deduplicate but preserve order.
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _iter_wordnet_candidates(
    wn,
    *,
    pos_list: Sequence[str],
    min_chars: int,
    exclude_words: set[str],
    exclude_substrings: bool,
) -> Iterable[str]:
    seen: set[str] = set()
    # Substring-exclude is a heuristic to prevent returning obvious morphological
    # expansions of input words (e.g., input "earth" -> "earthworm", "earthquake").
    exclude_roots = sorted(exclude_words, key=len, reverse=True) if exclude_substrings else []
    for pos in pos_list:
        for lemma in wn.all_lemma_names(pos=pos):
            if not isinstance(lemma, str):
                continue
            # WordNet uses underscores for multiword expressions; skip those.
            if "_" in lemma:
                continue
            raw = lemma.strip()
            if len(raw) < min_chars:
                continue
            if not _WORDLIKE_ASCII_RE.fullmatch(raw):
                continue
            lw = raw.lower()
            if lw in exclude_words:
                continue
            if exclude_roots and any(root in lw for root in exclude_roots):
                continue
            if lw in seen:
                continue
            seen.add(lw)
            yield raw


def _compute_all_but_the_top_transform(
    candidate_vecs: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    All-but-the-Top: Remove mean and top k principal components.
    
    Args:
        candidate_vecs: (N, D) array of candidate word vectors
        k: number of top principal components to remove
    
    Returns:
        mu: (D,) mean vector
        P_keep: (D, D) projection matrix that removes top k PCs
    """
    if k <= 0:
        # Return identity transform
        D = candidate_vecs.shape[1]
        return np.zeros(D, dtype=np.float32), np.eye(D, dtype=np.float32)
    
    # Center
    mu = candidate_vecs.mean(axis=0, dtype=np.float32)
    X_centered = candidate_vecs - mu
    
    # Compute covariance matrix and its eigenvectors
    # Use SVD on X_centered^T @ X_centered for numerical stability
    cov = (X_centered.T @ X_centered) / len(X_centered)
    
    # Get top k eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigvals)[::-1]
    eigvecs_sorted = eigvecs[:, idx]
    
    # Take top k principal directions
    k_actual = min(k, eigvecs_sorted.shape[1])
    U_k = eigvecs_sorted[:, :k_actual].astype(np.float32)
    
    # Projection matrix: I - U_k @ U_k^T
    P_remove = U_k @ U_k.T
    D = candidate_vecs.shape[1]
    P_keep = np.eye(D, dtype=np.float32) - P_remove
    
    return mu, P_keep


def _rank_top_wordnet_words_v2(
    embeddings: np.ndarray,
    tokenizer,
    group_mean_vectors: List[np.ndarray],
    group_weights: List[float],
    *,
    top_k: int,
    wn_pos: Sequence[str],
    min_chars: int,
    max_token_len: int,
    exclude_words: set[str],
    exclude_substrings: bool,
    candidate_batch: int,
    use_spherical_mean: bool,
    use_per_group_scoring: bool,
    abt_enabled: bool,
    abt_num_components: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    V2: Rank words using per-group scoring and All-but-the-Top transform.
    """
    wn = _load_wordnet()

    # Normalize group mean vectors for cosine similarity
    group_mean_vectors_normalized = []
    for gm in group_mean_vectors:
        gm_norm = float(np.linalg.norm(gm))
        if gm_norm > 1e-10:
            group_mean_vectors_normalized.append((gm / gm_norm).astype(np.float32))
        else:
            group_mean_vectors_normalized.append(gm.astype(np.float32))

    # Bucket candidates by token length
    buckets: DefaultDict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
    candidates_seen = 0
    candidates_kept = 0
    skipped_empty = 0
    skipped_too_long = 0

    for word in _iter_wordnet_candidates(
        wn,
        pos_list=wn_pos,
        min_chars=min_chars,
        exclude_words=exclude_words,
        exclude_substrings=exclude_substrings,
    ):
        candidates_seen += 1
        ids = _tokenize_no_special(tokenizer, word)
        if len(ids) == 0:
            skipped_empty += 1
            continue
        if len(ids) > max_token_len:
            skipped_too_long += 1
            continue
        buckets[len(ids)].append((word, ids))
        candidates_kept += 1

    if candidates_kept == 0:
        raise ValueError("No WordNet candidates left after filtering/tokenization")

    # First pass: collect ALL candidate vectors for All-but-the-Top transform
    all_candidate_vecs = []
    all_candidate_metadata = []  # (word, token_ids)
    
    for tok_len in sorted(buckets.keys()):
        items = buckets[tok_len]
        for start in range(0, len(items), candidate_batch):
            batch_items = items[start : start + candidate_batch]
            B = len(batch_items)
            token_id_mat = np.empty((B, tok_len), dtype=np.int32)
            
            for i, (w, ids) in enumerate(batch_items):
                token_id_mat[i, :] = np.array(ids, dtype=np.int32)
            
            # Compute candidate vectors with spherical averaging if enabled
            if use_spherical_mean:
                vecs_list = []
                for ids in [batch_items[i][1] for i in range(B)]:
                    vec = _mean_token_vectors_spherical(embeddings, ids)
                    vecs_list.append(vec)
                vecs = np.stack(vecs_list, axis=0)
            else:
                vecs = embeddings[token_id_mat].mean(axis=1, dtype=np.float32)
            
            all_candidate_vecs.append(vecs)
            all_candidate_metadata.extend(batch_items)
    
    all_candidate_vecs = np.vstack(all_candidate_vecs)
    
    # Apply All-but-the-Top transform if enabled
    if abt_enabled and abt_num_components > 0:
        mu, P_keep = _compute_all_but_the_top_transform(all_candidate_vecs, abt_num_components)
        # Transform: v' = P_keep @ (v - mu)
        all_candidate_vecs_transformed = (all_candidate_vecs - mu) @ P_keep.T
        # Also transform group vectors
        group_mean_vectors_transformed = []
        for gm in group_mean_vectors_normalized:
            gm_transformed = (gm - mu) @ P_keep.T
            # Renormalize after transform
            gm_transformed = _normalize_vector(gm_transformed)
            group_mean_vectors_transformed.append(gm_transformed)
        group_mean_vectors_for_scoring = group_mean_vectors_transformed
        candidate_vecs_for_scoring = all_candidate_vecs_transformed
    else:
        group_mean_vectors_for_scoring = group_mean_vectors_normalized
        candidate_vecs_for_scoring = all_candidate_vecs

    # Compute scores
    import heapq
    heap: List[Tuple[float, str, List[int]]] = []
    
    if use_per_group_scoring:
        # Per-group scoring: score(w) = sum_g w_g * cos(v_w, m_g)
        for i, (word, token_ids) in enumerate(all_candidate_metadata):
            v_w = candidate_vecs_for_scoring[i]
            v_w_norm = float(np.linalg.norm(v_w))
            
            if v_w_norm <= 1e-10:
                continue
            
            v_w_normalized = v_w / v_w_norm
            
            score = 0.0
            for g_idx, (gm, gw) in enumerate(zip(group_mean_vectors_for_scoring, group_weights)):
                gm_norm = float(np.linalg.norm(gm))
                if gm_norm > 1e-10:
                    cos_sim = float(np.dot(v_w_normalized, gm) / gm_norm)
                    score += gw * cos_sim
            
            if len(heap) < top_k:
                heapq.heappush(heap, (score, word, token_ids))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, word, token_ids))
    else:
        # Original single-essence-vector scoring
        # Compute overall essence vector as weighted average
        weighted_sum = np.zeros_like(group_mean_vectors_for_scoring[0])
        for gm, gw in zip(group_mean_vectors_for_scoring, group_weights):
            weighted_sum += gw * gm
        essence_vec = weighted_sum / len(group_mean_vectors_for_scoring)
        essence_norm = float(np.linalg.norm(essence_vec))
        
        if essence_norm <= 1e-10:
            raise ValueError("Essence vector has zero norm")
        
        essence_vec_normalized = essence_vec / essence_norm
        
        for i, (word, token_ids) in enumerate(all_candidate_metadata):
            v_w = candidate_vecs_for_scoring[i]
            v_w_norm = float(np.linalg.norm(v_w))
            
            if v_w_norm <= 1e-10:
                continue
            
            v_w_normalized = v_w / v_w_norm
            score = float(np.dot(v_w_normalized, essence_vec_normalized))
            
            if len(heap) < top_k:
                heapq.heappush(heap, (score, word, token_ids))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, word, token_ids))

    # Sort and format results
    best = sorted(heap, key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, w, ids in best:
        try:
            token_pieces = [tokenizer.decode([tid]) for tid in ids]
        except Exception:
            token_pieces = []
        out.append(
            {
                "word": w,
                "cosine_similarity": float(score) if use_per_group_scoring else float(score),
                "score": float(score),  # Keep both for compatibility
                "token_ids": [int(t) for t in ids],
                "token_pieces": token_pieces,
            }
        )

    stats = {
        "wordnet_pos": list(wn_pos),
        "min_word_chars": int(min_chars),
        "max_token_len": int(max_token_len),
        "exclude_words_count": int(len(exclude_words)),
        "exclude_substrings": bool(exclude_substrings),
        "candidates_seen": int(candidates_seen),
        "candidates_kept": int(candidates_kept),
        "skipped_empty": int(skipped_empty),
        "skipped_too_long": int(skipped_too_long),
        "v2_features": {
            "use_spherical_mean": bool(use_spherical_mean),
            "use_per_group_scoring": bool(use_per_group_scoring),
            "abt_enabled": bool(abt_enabled),
            "abt_num_components": int(abt_num_components) if abt_enabled else 0,
        }
    }
    return out, stats


def compute_essence_wordnet_v2(
    doc: Dict[str, Any],
    *,
    repo_id_override: Optional[str] = None,
    top_k_words: int = DEFAULT_TOP_K_WORDS,
    wordnet_pos: str = DEFAULT_WORDNET_POS,
    min_word_chars: int = DEFAULT_MIN_WORD_CHARS,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
    candidate_batch: int = DEFAULT_CANDIDATE_BATCH,
    exclude_input: bool = DEFAULT_EXCLUDE_INPUT,
    exclude_component_words: bool = DEFAULT_EXCLUDE_COMPONENT_WORDS,
    exclude_substrings: bool = DEFAULT_EXCLUDE_SUBSTRINGS,
    weight_clip_abs: Optional[float] = DEFAULT_WEIGHT_CLIP,
    use_spherical_mean: bool = DEFAULT_USE_SPHERICAL_MEAN,
    use_per_group_scoring: bool = DEFAULT_USE_PER_GROUP_SCORING,
    abt_enabled: bool = DEFAULT_ABT_ENABLED,
    abt_num_components: int = DEFAULT_ABT_NUM_COMPONENTS,
) -> Dict[str, Any]:
    """
    V2: Compute essence with three mathematical improvements:
    1. Spherical averaging
    2. Per-group scoring
    3. All-but-the-Top anisotropy correction
    """
    _validate_input(doc)

    repo_id = (
        repo_id_override
        or (doc.get("repo_id") if isinstance(doc.get("repo_id"), str) else None)
        or DEFAULT_REPO_ID
    )

    tensor_name, shard_name = extract_embeddings.find_embedding_shard(repo_id)
    shard_path, _cache_dir = extract_embeddings.download_files(repo_id, shard_name)
    embeddings = extract_embeddings.load_embedding_matrix(shard_path, tensor_name)

    vocab_size, dim = embeddings.shape
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    group_results, group_mean_vectors, group_weights, warnings, any_weight_clipped = (
        _compute_essence_vector_v2(
            doc, embeddings, tokenizer,
            weight_clip_abs=weight_clip_abs,
            use_spherical_mean=use_spherical_mean
        )
    )

    exclude_words = _build_exclude_word_set(
        doc["groups"],
        exclude_input=exclude_input,
        exclude_component_words=exclude_component_words,
    )

    wn_pos_list = _parse_pos_list(wordnet_pos)
    top_words, wordnet_stats = _rank_top_wordnet_words_v2(
        embeddings=embeddings,
        tokenizer=tokenizer,
        group_mean_vectors=group_mean_vectors,
        group_weights=group_weights,
        top_k=int(top_k_words),
        wn_pos=wn_pos_list,
        min_chars=int(min_word_chars),
        max_token_len=int(max_token_len),
        exclude_words=exclude_words,
        exclude_substrings=bool(exclude_substrings),
        candidate_batch=int(candidate_batch),
        use_spherical_mean=use_spherical_mean,
        use_per_group_scoring=use_per_group_scoring,
        abt_enabled=abt_enabled,
        abt_num_components=abt_num_components,
    )

    # Compute overall essence vector for metadata (even if not used for scoring)
    if use_per_group_scoring:
        # For metadata only: compute weighted average
        weighted_sum = np.zeros_like(group_mean_vectors[0])
        for gm, gw in zip(group_mean_vectors, group_weights):
            weighted_sum += gw * gm
        overall_vec = weighted_sum / len(group_mean_vectors)
    else:
        weighted_group_vectors = [gm * gw for gm, gw in zip(group_mean_vectors, group_weights)]
        overall_vec = np.stack(weighted_group_vectors, axis=0).mean(axis=0, dtype=np.float32)

    out_doc = copy.deepcopy(doc)
    out_doc["_output"] = {
        "model": {
            "repo_id": repo_id,
            "embedding_tensor": tensor_name or "auto-detected",
            "vocab_size": int(vocab_size),
            "embedding_dim": int(dim),
        },
        "settings": {
            "version": "v2",
            "entry_vector_rule": "spherical_mean" if use_spherical_mean else "mean_tokens",
            "group_aggregation": "spherical_mean" if use_spherical_mean else "mean_entries",
            "overall_aggregation": "per_group_scoring" if use_per_group_scoring else "mean_weighted_groups",
            "top_k_words": int(top_k_words),
            "wordnet_pos": ",".join(wn_pos_list),
            "min_word_chars": int(min_word_chars),
            "max_token_len": int(max_token_len),
            "candidate_batch": int(candidate_batch),
            "exclude_input": bool(exclude_input),
            "exclude_component_words": bool(exclude_component_words),
            "exclude_substrings": bool(exclude_substrings),
            "weight_clip_abs": None if weight_clip_abs is None else float(weight_clip_abs),
            "any_weight_clipped": bool(any_weight_clipped),
            "use_spherical_mean": bool(use_spherical_mean),
            "use_per_group_scoring": bool(use_per_group_scoring),
            "abt_enabled": bool(abt_enabled),
            "abt_num_components": int(abt_num_components) if abt_enabled else 0,
        },
        "groups": group_results,
        "overall": {"norm": float(np.linalg.norm(overall_vec))},
        "top_words": top_words,
        "wordnet_stats": wordnet_stats,
        "warnings": warnings,
    }

    return out_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2: Compute word-group essence with spherical averaging, per-group scoring, and All-but-the-Top."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write output JSON (otherwise prints to stdout)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"Override model repo_id (default: JSON repo_id or {DEFAULT_REPO_ID})",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_WORDS, help="Top-k words to return")
    parser.add_argument(
        "--wordnet-pos",
        default=DEFAULT_WORDNET_POS,
        help="Comma-separated WordNet POS tags (default: n,v). Options: n,v,a,r,s",
    )
    parser.add_argument("--min-word-chars", type=int, default=DEFAULT_MIN_WORD_CHARS)
    parser.add_argument("--max-token-len", type=int, default=DEFAULT_MAX_TOKEN_LEN)
    parser.add_argument("--candidate-batch", type=int, default=DEFAULT_CANDIDATE_BATCH)
    parser.add_argument("--no-exclude-input", action="store_true")
    parser.add_argument("--no-exclude-component-words", action="store_true")
    parser.add_argument("--no-exclude-substrings", action="store_true")
    parser.add_argument("--weight-clip-abs", type=float, default=DEFAULT_WEIGHT_CLIP)
    parser.add_argument("--no-weight-clip", action="store_true")
    
    # V2 options
    parser.add_argument("--no-spherical-mean", action="store_true", help="Disable spherical averaging")
    parser.add_argument("--no-per-group-scoring", action="store_true", help="Disable per-group scoring")
    parser.add_argument("--no-abt", action="store_true", help="Disable All-but-the-Top transform")
    parser.add_argument("--abt-components", type=int, default=DEFAULT_ABT_NUM_COMPONENTS, help="Number of PCs to remove (default: 5)")

    args = parser.parse_args()

    input_path = Path(args.input)
    try:
        with input_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read input JSON: {e}", file=sys.stderr)
        sys.exit(1)

    weight_clip_abs: Optional[float]
    if args.no_weight_clip:
        weight_clip_abs = None
    else:
        weight_clip_abs = float(args.weight_clip_abs)
        if weight_clip_abs <= 0:
            weight_clip_abs = None

    try:
        out = compute_essence_wordnet_v2(
            doc=doc,
            repo_id_override=args.repo_id,
            top_k_words=int(args.top_k),
            wordnet_pos=str(args.wordnet_pos),
            min_word_chars=int(args.min_word_chars),
            max_token_len=int(args.max_token_len),
            candidate_batch=int(args.candidate_batch),
            exclude_input=not bool(args.no_exclude_input),
            exclude_component_words=not bool(args.no_exclude_component_words),
            exclude_substrings=not bool(args.no_exclude_substrings),
            weight_clip_abs=weight_clip_abs,
            use_spherical_mean=not bool(args.no_spherical_mean),
            use_per_group_scoring=not bool(args.no_per_group_scoring),
            abt_enabled=not bool(args.no_abt),
            abt_num_components=int(args.abt_components),
        )
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    out_json = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(out_json, encoding="utf-8")
        print(f"✅ Wrote output JSON to: {out_path}")
    else:
        print(out_json)


if __name__ == "__main__":
    main()

