#!/usr/bin/env python3
"""
V2.1: Build on v2 with two highest-ROI search-quality improvements:

1) Word frequency filtering (WordNet ∩ wordfreq)
   - Filters WordNet lemma candidates to common English words using `wordfreq`.
   - Default: keep words in `wordfreq.top_n_list("en", 200000)`.

2) Tokenization-invariant embedding
   - v(text) = normalize(0.5 * (embed(text) + embed(" " + text)))
   - Applied to both input entries and candidate words.

This keeps all v2 improvements:
- Spherical averaging
- Per-group scoring
- All-but-the-Top anisotropy correction
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
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

# V2 defaults (still apply)
DEFAULT_USE_SPHERICAL_MEAN = True
DEFAULT_USE_PER_GROUP_SCORING = True
DEFAULT_ABT_ENABLED = True
DEFAULT_ABT_NUM_COMPONENTS = 5

# V2.1 defaults
DEFAULT_FREQ_FILTER_ENABLED = True
DEFAULT_FREQ_TOP_N = 200_000
DEFAULT_FREQ_MIN_ZIPF: Optional[float] = None
DEFAULT_TOKENIZATION_INVARIANCE = True

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
    token_ids = list(token_ids)
    if len(token_ids) == 0:
        raise ValueError("Tokenized to zero tokens")
    vecs = embeddings[token_ids]  # (T, D)
    return vecs.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)


def _mean_token_vectors_spherical(embeddings: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    """
    Spherical mean: normalize each token vector, average, then normalize result.
    """
    token_ids = list(token_ids)
    if len(token_ids) == 0:
        raise ValueError("Tokenized to zero tokens")

    vecs = embeddings[token_ids]  # (T, D)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    vecs_normalized = vecs / norms

    mean_vec = vecs_normalized.mean(axis=0, dtype=np.float32)
    return _normalize_vector(mean_vec)


def _text_vector(
    embeddings: np.ndarray,
    tokenizer,
    text: str,
    *,
    use_spherical_mean: bool,
    tokenization_invariance: bool,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Compute a text vector, optionally using tokenization invariance:
        v(text) = normalize(0.5*(embed(text) + embed(" " + text)))

    Returns:
        vec, token_ids, token_ids_prefixed
    """
    token_ids = _tokenize_no_special(tokenizer, text)
    if len(token_ids) == 0:
        raise ValueError("Tokenized to zero tokens")

    if use_spherical_mean:
        v = _mean_token_vectors_spherical(embeddings, token_ids)
    else:
        v = _mean_token_vectors(embeddings, token_ids)

    token_ids_prefixed: List[int] = []
    if tokenization_invariance:
        token_ids_prefixed = _tokenize_no_special(tokenizer, " " + text)
        if len(token_ids_prefixed) > 0:
            if use_spherical_mean:
                v2 = _mean_token_vectors_spherical(embeddings, token_ids_prefixed)
            else:
                v2 = _mean_token_vectors(embeddings, token_ids_prefixed)
            v = _normalize_vector(0.5 * (v + v2))
        else:
            # Fall back to non-prefixed if " "+text tokenizes to empty
            token_ids_prefixed = []
            v = _normalize_vector(v) if use_spherical_mean else v.astype(np.float32, copy=False)

    return v.astype(np.float32, copy=False), token_ids, token_ids_prefixed


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


@lru_cache(maxsize=8)
def _wordfreq_top_set(n: int) -> frozenset[str]:
    """
    Cached set of top-N English words by frequency.
    """
    from wordfreq import top_n_list

    # wordfreq returns lowercase words; normalize just in case.
    return frozenset(w.lower() for w in top_n_list("en", int(n)))


def _passes_freq_filter(
    word_lc: str,
    *,
    enabled: bool,
    top_n: int,
    min_zipf: Optional[float],
) -> bool:
    if not enabled:
        return True
    if min_zipf is not None:
        from wordfreq import zipf_frequency

        return float(zipf_frequency(word_lc, "en")) >= float(min_zipf)
    # Default: membership in top-N list
    return word_lc in _wordfreq_top_set(int(top_n))


def _compute_essence_vector_v21(
    doc: Dict[str, Any],
    embeddings: np.ndarray,
    tokenizer,
    *,
    weight_clip_abs: Optional[float],
    use_spherical_mean: bool,
    tokenization_invariance: bool,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[float], List[str], bool]:
    """
    V2.1: Same as v2 group computation, but entry vectors optionally use
    tokenization invariance.
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

            try:
                entry_vec, token_ids, token_ids_prefixed = _text_vector(
                    embeddings,
                    tokenizer,
                    entry,
                    use_spherical_mean=use_spherical_mean,
                    tokenization_invariance=tokenization_invariance,
                )
            except Exception:
                warnings.append(f"Group[{gi}].entries[{ei}] tokenized to zero tokens; skipping")
                continue

            try:
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
            except Exception:
                tokens = []

            entry_vecs.append(entry_vec)
            entry_info: Dict[str, Any] = {
                "text": entry,
                "token_ids": [int(t) for t in token_ids],
                "tokens": tokens,
            }
            if tokenization_invariance and token_ids_prefixed:
                entry_info["token_ids_prefixed"] = [int(t) for t in token_ids_prefixed]
            entry_infos.append(entry_info)

        if len(entry_vecs) == 0:
            raise ValueError(f"Group[{gi}] has zero valid entries after tokenization")

        # Compute group mean with optional spherical averaging
        if use_spherical_mean:
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

        group_mean_vectors.append(group_mean)
        group_weights.append(weight_used)

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
    try:
        import nltk  # noqa: F401
        from nltk.corpus import wordnet as wn
    except Exception as e:
        raise RuntimeError("WordNet mode requires NLTK. Install it with `pip install nltk`.") from e

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


def _iter_wordnet_candidates_v21(
    wn,
    *,
    pos_list: Sequence[str],
    min_chars: int,
    exclude_words: set[str],
    exclude_substrings: bool,
    freq_filter_enabled: bool,
    freq_top_n: int,
    freq_min_zipf: Optional[float],
) -> Iterable[str]:
    seen: set[str] = set()
    exclude_roots = sorted(exclude_words, key=len, reverse=True) if exclude_substrings else []
    for pos in pos_list:
        for lemma in wn.all_lemma_names(pos=pos):
            if not isinstance(lemma, str):
                continue
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
            if not _passes_freq_filter(
                lw,
                enabled=bool(freq_filter_enabled),
                top_n=int(freq_top_n),
                min_zipf=freq_min_zipf,
            ):
                continue
            seen.add(lw)
            yield raw


def _compute_abt_stats(candidate_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute All-but-the-Top parameters:
      - mu (mean vector)
      - U_k (top principal directions), or None if k<=0
    """
    mu = candidate_vecs.mean(axis=0, dtype=np.float32)
    if k <= 0:
        return mu, None

    X = candidate_vecs - mu
    cov = (X.T @ X) / len(X)  # (D,D)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs_sorted = eigvecs[:, idx]
    k_actual = min(int(k), eigvecs_sorted.shape[1])
    U_k = eigvecs_sorted[:, :k_actual].astype(np.float32, copy=False)
    return mu.astype(np.float32, copy=False), U_k


def _apply_abt(X: np.ndarray, U_k: Optional[np.ndarray]) -> np.ndarray:
    """
    Apply ABT projection to already-centered vectors X:
      X' = X - (X @ U_k) @ U_k.T
    """
    if U_k is None:
        return X
    return X - (X @ U_k) @ U_k.T


def _rank_top_wordnet_words_v21(
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
    tokenization_invariance: bool,
    freq_filter_enabled: bool,
    freq_top_n: int,
    freq_min_zipf: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    wn = _load_wordnet()

    # Normalize group mean vectors for cosine similarity
    group_mean_vectors_normalized: List[np.ndarray] = []
    for gm in group_mean_vectors:
        group_mean_vectors_normalized.append(_normalize_vector(gm))

    # Bucket candidates by token length (but store ids and prefixed ids if needed)
    buckets: DefaultDict[int, List[Tuple[str, List[int], List[int]]]] = defaultdict(list)
    candidates_seen = 0
    candidates_kept = 0
    skipped_empty = 0
    skipped_too_long = 0

    for word in _iter_wordnet_candidates_v21(
        wn,
        pos_list=wn_pos,
        min_chars=min_chars,
        exclude_words=exclude_words,
        exclude_substrings=exclude_substrings,
        freq_filter_enabled=freq_filter_enabled,
        freq_top_n=freq_top_n,
        freq_min_zipf=freq_min_zipf,
    ):
        candidates_seen += 1
        ids = _tokenize_no_special(tokenizer, word)
        if len(ids) == 0:
            skipped_empty += 1
            continue
        if len(ids) > max_token_len:
            skipped_too_long += 1
            continue
        ids_prefixed: List[int] = []
        if tokenization_invariance:
            ids_prefixed = _tokenize_no_special(tokenizer, " " + word)
        buckets[len(ids)].append((word, ids, ids_prefixed))
        candidates_kept += 1

    if candidates_kept == 0:
        raise ValueError("No WordNet candidates left after filtering/tokenization")

    # Collect candidate vectors (for ABT)
    all_candidate_vecs: List[np.ndarray] = []
    all_candidate_metadata: List[Tuple[str, List[int], List[int]]] = []

    for tok_len in sorted(buckets.keys()):
        items = buckets[tok_len]
        for start in range(0, len(items), candidate_batch):
            batch_items = items[start : start + candidate_batch]
            vecs_list: List[np.ndarray] = []
            for w, ids, ids_prefixed in batch_items:
                if tokenization_invariance:
                    v1 = (
                        _mean_token_vectors_spherical(embeddings, ids)
                        if use_spherical_mean
                        else _mean_token_vectors(embeddings, ids)
                    )
                    if ids_prefixed:
                        v2 = (
                            _mean_token_vectors_spherical(embeddings, ids_prefixed)
                            if use_spherical_mean
                            else _mean_token_vectors(embeddings, ids_prefixed)
                        )
                        v = _normalize_vector(0.5 * (v1 + v2))
                    else:
                        v = _normalize_vector(v1) if use_spherical_mean else v1.astype(np.float32, copy=False)
                else:
                    v = (
                        _mean_token_vectors_spherical(embeddings, ids)
                        if use_spherical_mean
                        else _mean_token_vectors(embeddings, ids)
                    )
                vecs_list.append(v.astype(np.float32, copy=False))
            vecs = np.stack(vecs_list, axis=0)
            all_candidate_vecs.append(vecs)
            all_candidate_metadata.extend(batch_items)

    candidate_vecs = np.vstack(all_candidate_vecs).astype(np.float32, copy=False)  # (N,D)

    # Center + ABT (if enabled)
    mu = candidate_vecs.mean(axis=0, dtype=np.float32)
    U_k: Optional[np.ndarray] = None
    if abt_enabled and abt_num_components > 0:
        mu, U_k = _compute_abt_stats(candidate_vecs, int(abt_num_components))

    X = (candidate_vecs - mu).astype(np.float32, copy=False)
    X = _apply_abt(X, U_k)

    # Transform group vectors consistently
    group_for_scoring: List[np.ndarray] = []
    for gm in group_mean_vectors_normalized:
        gm_c = (gm - mu).astype(np.float32, copy=False)
        gm_c = _apply_abt(gm_c.reshape(1, -1), U_k).reshape(-1)
        group_for_scoring.append(_normalize_vector(gm_c))

    import heapq

    heap: List[Tuple[float, str, List[int]]] = []

    if use_per_group_scoring:
        for i, (word, token_ids, _token_ids_prefixed) in enumerate(all_candidate_metadata):
            v = X[i]
            v_norm = float(np.linalg.norm(v))
            if v_norm <= 1e-10:
                continue
            v_unit = v / v_norm
            score = 0.0
            for gm, gw in zip(group_for_scoring, group_weights):
                gm_norm = float(np.linalg.norm(gm))
                if gm_norm > 1e-10:
                    score += float(gw) * float(np.dot(v_unit, gm) / gm_norm)
            if len(heap) < top_k:
                heapq.heappush(heap, (score, word, token_ids))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, word, token_ids))
    else:
        weighted_sum = np.zeros_like(group_for_scoring[0])
        for gm, gw in zip(group_for_scoring, group_weights):
            weighted_sum += float(gw) * gm
        essence = weighted_sum / len(group_for_scoring)
        essence = _normalize_vector(essence)
        for i, (word, token_ids, _token_ids_prefixed) in enumerate(all_candidate_metadata):
            v = X[i]
            v_norm = float(np.linalg.norm(v))
            if v_norm <= 1e-10:
                continue
            v_unit = v / v_norm
            score = float(np.dot(v_unit, essence))
            if len(heap) < top_k:
                heapq.heappush(heap, (score, word, token_ids))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, word, token_ids))

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
                "cosine_similarity": float(score),
                "score": float(score),
                "token_ids": [int(t) for t in ids],
                "token_pieces": token_pieces,
            }
        )

    stats: Dict[str, Any] = {
        "wordnet_pos": list(wn_pos),
        "min_word_chars": int(min_chars),
        "max_token_len": int(max_token_len),
        "exclude_words_count": int(len(exclude_words)),
        "exclude_substrings": bool(exclude_substrings),
        "candidates_seen": int(candidates_seen),
        "candidates_kept": int(candidates_kept),
        "skipped_empty": int(skipped_empty),
        "skipped_too_long": int(skipped_too_long),
        "v21_features": {
            "use_spherical_mean": bool(use_spherical_mean),
            "use_per_group_scoring": bool(use_per_group_scoring),
            "abt_enabled": bool(abt_enabled),
            "abt_num_components": int(abt_num_components) if abt_enabled else 0,
            "tokenization_invariance": bool(tokenization_invariance),
            "freq_filter_enabled": bool(freq_filter_enabled),
            "freq_top_n": int(freq_top_n) if freq_filter_enabled else None,
            "freq_min_zipf": None if freq_min_zipf is None else float(freq_min_zipf),
        },
    }
    return out, stats


def compute_essence_wordnet_v21(
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
    tokenization_invariance: bool = DEFAULT_TOKENIZATION_INVARIANCE,
    freq_filter_enabled: bool = DEFAULT_FREQ_FILTER_ENABLED,
    freq_top_n: int = DEFAULT_FREQ_TOP_N,
    freq_min_zipf: Optional[float] = DEFAULT_FREQ_MIN_ZIPF,
) -> Dict[str, Any]:
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
        _compute_essence_vector_v21(
            doc,
            embeddings,
            tokenizer,
            weight_clip_abs=weight_clip_abs,
            use_spherical_mean=use_spherical_mean,
            tokenization_invariance=tokenization_invariance,
        )
    )

    exclude_words = _build_exclude_word_set(
        doc["groups"],
        exclude_input=exclude_input,
        exclude_component_words=exclude_component_words,
    )

    wn_pos_list = _parse_pos_list(wordnet_pos)
    top_words, wordnet_stats = _rank_top_wordnet_words_v21(
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
        tokenization_invariance=tokenization_invariance,
        freq_filter_enabled=freq_filter_enabled,
        freq_top_n=int(freq_top_n),
        freq_min_zipf=freq_min_zipf,
    )

    # Metadata overall vector (not necessarily used for scoring)
    if use_per_group_scoring:
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
            "version": "v2.1",
            "entry_vector_rule": (
                "tokenization_invariant+spherical_mean"
                if tokenization_invariance and use_spherical_mean
                else ("tokenization_invariant+mean_tokens" if tokenization_invariance else ("spherical_mean" if use_spherical_mean else "mean_tokens"))
            ),
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
            "tokenization_invariance": bool(tokenization_invariance),
            "freq_filter_enabled": bool(freq_filter_enabled),
            "freq_top_n": int(freq_top_n) if freq_filter_enabled else None,
            "freq_min_zipf": None if freq_min_zipf is None else float(freq_min_zipf),
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
        description="V2.1: v2 + (wordfreq filtering) + (tokenization invariance)."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_WORDS)
    parser.add_argument("--wordnet-pos", default=DEFAULT_WORDNET_POS)
    parser.add_argument("--min-word-chars", type=int, default=DEFAULT_MIN_WORD_CHARS)
    parser.add_argument("--max-token-len", type=int, default=DEFAULT_MAX_TOKEN_LEN)
    parser.add_argument("--candidate-batch", type=int, default=DEFAULT_CANDIDATE_BATCH)
    parser.add_argument("--no-exclude-input", action="store_true")
    parser.add_argument("--no-exclude-component-words", action="store_true")
    parser.add_argument("--no-exclude-substrings", action="store_true")
    parser.add_argument("--weight-clip-abs", type=float, default=DEFAULT_WEIGHT_CLIP)
    parser.add_argument("--no-weight-clip", action="store_true")

    # v2 options
    parser.add_argument("--no-spherical-mean", action="store_true")
    parser.add_argument("--no-per-group-scoring", action="store_true")
    parser.add_argument("--no-abt", action="store_true")
    parser.add_argument("--abt-components", type=int, default=DEFAULT_ABT_NUM_COMPONENTS)

    # v2.1 options
    parser.add_argument("--no-tokenization-invariance", action="store_true")
    parser.add_argument("--no-freq-filter", action="store_true")
    parser.add_argument("--freq-top-n", type=int, default=DEFAULT_FREQ_TOP_N)
    parser.add_argument("--freq-min-zipf", type=float, default=None)

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
        out = compute_essence_wordnet_v21(
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
            tokenization_invariance=not bool(args.no_tokenization_invariance),
            freq_filter_enabled=not bool(args.no_freq_filter),
            freq_top_n=int(args.freq_top_n),
            freq_min_zipf=args.freq_min_zipf,
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


