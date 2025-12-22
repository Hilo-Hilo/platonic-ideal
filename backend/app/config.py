from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    id: str
    repo_id: str
    description: str
    name: str  # Display name for UI
    speed: str # Subjective speed label for UI


DEFAULT_MODEL = ModelConfig(
    id="tinyllama-1.1b",
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    description="TinyLlama 1.1B (best balance of quality/speed from tests)",
    name="TinyLlama 1.1B (Recommended)",
    speed="fast",
)

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    DEFAULT_MODEL.id: DEFAULT_MODEL,
    "qwen-0.5b": ModelConfig(
        id="qwen-0.5b",
        repo_id="Qwen/Qwen2.5-0.5B",
        description="Qwen 0.5B baseline",
        name="Qwen 0.5B",
        speed="fast",
    ),
    "qwen-1.5b": ModelConfig(
        id="qwen-1.5b",
        repo_id="Qwen/Qwen2.5-1.5B",
        description="Qwen 1.5B larger embeddings",
        name="Qwen 1.5B",
        speed="medium",
    ),
    "qwen-3b": ModelConfig(
        id="qwen-3b",
        repo_id="Qwen/Qwen2.5-3B",
        description="Qwen 3B bilingual sharded",
        name="Qwen 3B",
        speed="slow",
    ),
    "qwen-7b": ModelConfig(
        id="qwen-7b",
        repo_id="Qwen/Qwen2.5-7B",
        description="Qwen 7B large bilingual (slow, high quality)",
        name="Qwen 7B (Large)",
        speed="very slow",
    ),
    "phi-2": ModelConfig(
        id="phi-2",
        repo_id="microsoft/phi-2",
        description="Microsoft Phi-2 2.7B reasoning-focused",
        name="Phi-2 2.7B",
        speed="slow",
    ),
    "gemma-2b": ModelConfig(
        id="gemma-2b",
        repo_id="google/gemma-2b",
        description="Google Gemma 2B lightweight",
        name="Gemma 2B",
        speed="medium",
    ),
}


