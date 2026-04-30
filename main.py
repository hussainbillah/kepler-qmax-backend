"""
FastAPI server for Kepler Q-Max QAOA Model
Run locally:  uvicorn main:app --reload --port 8000
Deploy:       Render / Railway / HuggingFace Spaces / Fly.io
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from hybrid_deployment_bridge import HybridModelBridge

app = FastAPI(
    title="Kepler Q-Max Quantum Inference API",
    description="QAOA p=10 ansatz, 25 qubits, trained on Sonicium Quantum Lab",
    version="1.0.0",
)

# CORS — allow your Lovable preview & published URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Lovable URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("KEPLER_API_KEY", "")  # Set this in Render env vars

# Pre-load model on startup (saves cold-start time)
_models = {}


def get_model(backend: str) -> HybridModelBridge:
    if backend not in _models:
        _models[backend] = HybridModelBridge(backend=backend)
    return _models[backend]


def verify_key(authorization: Optional[str]):
    if not API_KEY:
        return  # auth disabled
    if not authorization or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ────────── Schemas ──────────
class PredictRequest(BaseModel):
    input: List[float] = Field(..., description="Input vector, will be padded/truncated to 25", min_length=1, max_length=512)
    backend: str = Field(default="pennylane", pattern="^(pennylane|pytorch|qiskit)$")


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    backend: str
    expectation_values: Optional[List[float]] = None
    logits: Optional[List[float]] = None
    top_bitstring: Optional[str] = None


# ────────── Routes ──────────
@app.get("/")
def root():
    return {
        "service": "Kepler Q-Max Quantum Inference",
        "model": "QAOA p=10",
        "n_qubits": 25,
        "backends": ["pennylane", "pytorch", "qiskit"],
        "endpoints": ["/predict", "/compare", "/circuit", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, authorization: Optional[str] = Header(default=None)):
    verify_key(authorization)
    try:
        model = get_model(req.backend)
        result = model.predict(req.input)
        return PredictResponse(
            prediction=int(result.get("prediction", 0)),
            confidence=float(result.get("confidence", 0.0)),
            backend=result.get("backend", req.backend),
            expectation_values=result.get("expectation_values"),
            logits=result.get("logits"),
            top_bitstring=result.get("top_bitstring"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/compare")
def compare(req: PredictRequest, authorization: Optional[str] = Header(default=None)):
    verify_key(authorization)
    bridge = HybridModelBridge(backend=req.backend)
    return bridge.compare_backends(req.input)


@app.get("/circuit")
def circuit(authorization: Optional[str] = Header(default=None)):
    verify_key(authorization)
    try:
        model = get_model("pennylane")
        return model.model.get_circuit_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
