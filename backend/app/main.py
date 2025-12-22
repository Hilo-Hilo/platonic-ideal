from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import MODEL_REGISTRY, DEFAULT_MODEL
from backend.app.compute import compute_essence_payload
from backend.app.models import ComputeEssenceRequest, HealthResponse, ModelInfo
from backend.app.session_lock import SessionBusyError, session_lock

import os

# Filter models based on ALLOWED_MODELS env var
# Default to only small models if not specified
_ALLOWED_MODELS_STR = os.getenv("ALLOWED_MODELS", "tinyllama-1.1b,qwen-0.5b")
ALLOWED_MODEL_IDS = set(m.strip() for m in _ALLOWED_MODELS_STR.split(",") if m.strip())

# Validate that allowed models exist in registry
ALLOWED_REGISTRY = {
    k: v for k, v in MODEL_REGISTRY.items() 
    if k in ALLOWED_MODEL_IDS
}

app = FastAPI(
    title="Word-Group Essence API",
    version="0.1.0",
    description="Compute word-group essence vectors and return nearest WordNet dictionary words.",
)

import os

# Enable CORS for frontend
frontend_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins + [
        "http://localhost:3001", 
        "http://localhost:3002", 
        "http://localhost:3003", 
        "http://localhost:3004",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Return only allowed models with full metadata
    models_info = [
        ModelInfo(
            id=m.id,
            name=m.name,
            repo_id=m.repo_id,
            speed=m.speed,
            description=m.description
        )
        for m in ALLOWED_REGISTRY.values()
    ]
    
    return HealthResponse(
        status="ok",
        default_model_id=DEFAULT_MODEL.id if DEFAULT_MODEL.id in ALLOWED_REGISTRY else list(ALLOWED_REGISTRY.keys())[0],
        available_models=models_info,
    )


@app.post("/compute-essence")
async def compute_essence(
    req: ComputeEssenceRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")

    # Normalize requested models (max 3)
    if req.model_ids is not None:
        model_ids = req.model_ids
    else:
        model_ids = [req.model_id or DEFAULT_MODEL.id]

    # Validate against allowed list
    for mid in model_ids:
        if mid not in ALLOWED_REGISTRY:
             raise HTTPException(status_code=403, detail=f"Model '{mid}' is not available on this server.")

    if len(model_ids) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 models per request")

    # Build payload dict (Pydantic → dict)
    payload = {
        "model_ids": model_ids,
        "groups": [g.dict() for g in req.groups],
        "options": req.options.dict(),
    }

    try:
        async with session_lock(x_session_id):
            result = await compute_essence_payload(payload)
        return JSONResponse(result)
    except SessionBusyError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


