from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import MODEL_REGISTRY, DEFAULT_MODEL
from backend.app.compute import compute_essence_payload
from backend.app.models import ComputeEssenceRequest, HealthResponse
from backend.app.session_lock import SessionBusyError, session_lock

app = FastAPI(
    title="Word-Group Essence API",
    version="0.1.0",
    description="Compute word-group essence vectors and return nearest WordNet dictionary words.",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        default_model_id=DEFAULT_MODEL.id,
        available_models=list(MODEL_REGISTRY.keys()),
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


