from typing import List, Optional

from pydantic import BaseModel, Field, conint, constr, validator


class WordGroup(BaseModel):
    name: Optional[str] = None
    weight: float = 1.0
    entries: List[constr(strip_whitespace=True, min_length=1, max_length=100)] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of words/phrases in the group (max 50, max 100 chars each)",
    )


class ComputeOptions(BaseModel):
    top_k: conint(ge=1, le=100) = 20
    wordnet_pos: str = "n,v"  # comma-separated POS
    exclude_input: bool = True
    exclude_substrings: bool = True
    min_word_chars: conint(ge=1, le=20) = 3
    max_token_len: conint(ge=1, le=16) = 6
    candidate_batch: conint(ge=256, le=32768) = 4096
    weight_clip_abs: Optional[float] = 32.0

    @validator("wordnet_pos")
    def validate_pos(cls, v: str) -> str:
        allowed = {"n", "v", "a", "r", "s"}
        parts = [p.strip().lower() for p in v.split(",") if p.strip()]
        if not parts:
            raise ValueError("wordnet_pos must include at least one POS tag")
        for p in parts:
            if p not in allowed:
                raise ValueError(f"Invalid POS tag '{p}'. Allowed: {sorted(allowed)}")
        return ",".join(parts)


class ComputeEssenceRequest(BaseModel):
    # Preferred: batch request (max 3 models)
    model_ids: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(
        None,
        max_items=3,
        description="Model ids to run (max 3). If omitted, falls back to model_id or default.",
    )
    # Backwards-compatible single model
    model_id: Optional[str] = Field(
        None, description="Single model id (deprecated). Use model_ids instead."
    )
    groups: List[WordGroup] = Field(..., min_items=1, max_items=10)
    options: ComputeOptions = Field(default_factory=ComputeOptions)


class HealthResponse(BaseModel):
    status: str
    default_model_id: str
    available_models: List[str]

