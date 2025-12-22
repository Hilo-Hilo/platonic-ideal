import time
from typing import Any, Dict, List, Tuple

from fastapi.concurrency import run_in_threadpool

from backend.app.config import MODEL_REGISTRY, DEFAULT_MODEL
import word_group_essence_wordnet as wge


async def compute_essence_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute essence using the WordNet-based method. Runs in a thread pool to avoid
    blocking the event loop.
    """
    model_ids: List[str] = payload["model_ids"]
    groups = payload["groups"]
    opts = payload["options"]

    # Validate model ids
    for mid in model_ids:
        if mid not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_id '{mid}'. Available: {list(MODEL_REGISTRY)}")

    def _run_one(model_id: str) -> Tuple[str, Dict[str, Any]]:
        repo_id = MODEL_REGISTRY[model_id].repo_id
        doc = {"repo_id": repo_id, "groups": groups}

        start = time.time()
        result = wge.compute_essence_wordnet(
            doc=doc,
            repo_id_override=None,
            top_k_words=opts["top_k"],
            wordnet_pos=opts["wordnet_pos"],
            min_word_chars=opts["min_word_chars"],
            max_token_len=opts["max_token_len"],
            candidate_batch=opts["candidate_batch"],
            exclude_input=opts["exclude_input"],
            exclude_component_words=True,
            exclude_substrings=opts["exclude_substrings"],
            weight_clip_abs=opts["weight_clip_abs"],
        )
        elapsed = time.time() - start
        result["_output"]["timing_s"] = elapsed
        result["_output"]["model"]["model_id"] = model_id
        return model_id, result

    # Run sequentially for stability (prevents loading multiple large models at once).
    results: Dict[str, Any] = {}
    per_model_timing: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    total_start = time.time()
    for model_id in model_ids:
        try:
            mid, res = await run_in_threadpool(_run_one, model_id)
            results[mid] = res
            per_model_timing[mid] = float(res["_output"].get("timing_s", 0.0))
        except Exception as e:
            errors[model_id] = str(e)

    total_elapsed = time.time() - total_start

    return {
        "model_ids": model_ids,
        "results": results,
        "errors": errors,
        "timing_s": total_elapsed,
        "per_model_timing_s": per_model_timing,
    }


def get_default_model_id() -> str:
    return DEFAULT_MODEL.id

