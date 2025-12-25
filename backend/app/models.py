from typing import List, Optional

from pydantic import BaseModel, Field, conint, constr, field_validator


class WordGroup(BaseModel):
    name: Optional[str] = None
    weight: float = 1.0
    entries: List[constr(strip_whitespace=True, min_length=1, max_length=100)] = Field(
        ...,
        min_length=1,
        max_length=50,
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

    @field_validator("wordnet_pos")
    @classmethod
    def validate_pos(cls, v: str) -> str:
        allowed = {"n", "v", "a", "r", "s"}
        parts = [p.strip().lower() for p in v.split(",") if p.strip()]
        if not parts:
            raise ValueError("wordnet_pos must include at least one POS tag")
        for p in parts:
            if p not in allowed:
                raise ValueError(f"Invalid POS tag '{p}'. Allowed: {sorted(allowed)}")
        return ",".join(parts)


class ComputeOptionsV2(ComputeOptions):
    """V2 options with mathematical improvements."""
    use_spherical_mean: bool = True
    use_per_group_scoring: bool = True
    abt_enabled: bool = True
    abt_num_components: conint(ge=1, le=20) = 5


class ComputeOptionsV21(ComputeOptionsV2):
    """V2.1 options: adds wordfreq filtering + tokenization invariance on top of v2."""
    freq_filter_enabled: bool = True
    freq_top_n: conint(ge=1000, le=500000) = 200000
    freq_min_zipf: Optional[float] = None  # Alternative to top_n
    tokenization_invariance: bool = True


class ComputeOptionsV22(ComputeOptionsV21):
    """V2.2 options: adds robust centers, sense reranking, diagonal whitening on top of v2.1."""
    robust_group_center: bool = True
    trim_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    sense_rerank_enabled: bool = True
    sense_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    sense_rerank_pool_multiplier: conint(ge=1, le=50) = 10
    diag_whiten_enabled: bool = True
    diag_whiten_eps: float = 1e-6


class ComputeEssenceRequest(BaseModel):
    # Preferred: batch request (max 3 models)
    model_ids: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(
        None,
        max_length=3,
        description="Model ids to run (max 3). If omitted, falls back to model_id or default.",
    )
    # Backwards-compatible single model
    model_id: Optional[str] = Field(
        None, description="Single model id (deprecated). Use model_ids instead."
    )
    groups: List[WordGroup] = Field(..., min_length=1, max_length=10)
    options: ComputeOptions = Field(default_factory=ComputeOptions)


class ComputeEssenceRequestV2(BaseModel):
    """V2 request with improved mathematical options."""
    # Preferred: batch request (max 3 models)
    model_ids: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(
        None,
        max_length=3,
        description="Model ids to run (max 3). If omitted, falls back to model_id or default.",
    )
    # Backwards-compatible single model
    model_id: Optional[str] = Field(
        None, description="Single model id (deprecated). Use model_ids instead."
    )
    groups: List[WordGroup] = Field(..., min_length=1, max_length=10)
    options: ComputeOptionsV2 = Field(default_factory=ComputeOptionsV2)


class ComputeEssenceRequestV21(BaseModel):
    """V2.1 request: frequency-filtered candidates + tokenization invariance."""
    model_ids: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(
        None,
        max_length=3,
        description="Model ids to run (max 3). If omitted, falls back to model_id or default.",
    )
    model_id: Optional[str] = Field(
        None, description="Single model id (deprecated). Use model_ids instead."
    )
    groups: List[WordGroup] = Field(..., min_length=1, max_length=10)
    options: ComputeOptionsV21 = Field(default_factory=ComputeOptionsV21)


class ComputeEssenceRequestV22(BaseModel):
    """V2.2 request: adds robust centers, sense reranking, diagonal whitening."""
    model_ids: Optional[List[constr(strip_whitespace=True, min_length=1)]] = Field(
        None,
        max_length=3,
        description="Model ids to run (max 3). If omitted, falls back to model_id or default.",
    )
    model_id: Optional[str] = Field(
        None, description="Single model id (deprecated). Use model_ids instead."
    )
    groups: List[WordGroup] = Field(..., min_length=1, max_length=10)
    options: ComputeOptionsV22 = Field(default_factory=ComputeOptionsV22)


class ModelInfo(BaseModel):
    id: str
    name: str
    repo_id: str
    speed: str
    description: str

class HealthResponse(BaseModel):
    status: str
    default_model_id: str
    available_models: List[ModelInfo]

