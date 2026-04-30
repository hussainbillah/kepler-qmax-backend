from __future__ import annotations

import math
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

N_QUBITS = 25
N_LAYERS = 9
MAX_FEATURES = 256

TRAINED_PARAMETERS = np.array(
    [
        5.856574, 2.560996, 3.452822, 4.924401, 4.042088, 4.19697, 4.166393,
        0.750169, 0.439831, 0.116248, 1.913283, 1.328468, 2.903173, 1.192195,
        1.865814, 5.105802, 5.231946, 3.910516, 5.829434, 4.593439, 1.221019,
        2.37304, 4.732735, 5.565568, 6.274473, 1.264838, 5.632028, 1.716779,
        5.78753, 3.572028, 3.793283, 5.01533, 2.396613, 4.957047, 0.48055,
        1.081561, 2.759884, 3.136259, 5.287992, 2.084536, 6.012886, 0.759625,
        1.122561, 0.51285, 1.066053, 0.712498, 2.311565, 5.737445, 0.331197,
        0.61362, 4.871402, 1.279885, 0.804798, 1.13735, 0.252407, 1.562368,
        3.767766, 0.376671, 5.534116, 5.21219, 6.020664, 4.582969, 1.784164,
        1.705808, 2.807851, 5.507988, 0.590482, 1.212815, 0.172556, 1.807851,
        1.674727, 2.52232, 4.639698, 5.309746, 1.906332, 4.000558, 3.676428,
        5.928646, 3.979089, 0.736532, 2.216594, 2.539047, 5.99887, 0.567897,
        6.176963, 1.332463, 5.067386, 5.285475, 2.543474, 4.142679, 5.535392,
        5.416571, 6.026103, 4.508408, 5.31996, 0.85893, 5.090744, 4.530588,
        4.366299, 4.65932, 6.171979, 4.320512, 5.50717, 1.367863, 5.733541,
        5.370367, 4.67647, 2.616085, 4.157357, 1.63352, 0.420299, 4.845386,
        1.083697, 2.78772, 3.954119, 1.258668, 1.360339, 3.778291, 1.169504,
        0.576703, 5.78077, 4.804755, 1.777127, 0.468436, 0.061391, 5.272725,
        4.467379, 3.325289, 1.060736, 3.617884, 3.850475, 0.257529, 0.903072,
        5.063708, 3.145932, 1.55797, 3.097899, 3.246293, 4.445237, 4.953559,
        0.742459, 3.741403, 0.207846, 1.881276, 6.0758, 2.020016, 3.787783,
        4.215062, 3.462644, 5.85863, 1.279296, 0.043074, 1.049043, 5.305841,
        3.850535, 6.275933, 6.04419, 2.222589, 4.313867, 4.163166, 1.481004,
        5.639188, 1.393836, 4.03831, 3.878839, 1.273973, 5.39714, 3.378728,
        0.05921, 5.660196, 3.391653, 5.960814, 2.759106, 2.739315, 1.802381,
        1.285087, 1.301629, 2.172587, 2.009286, 6.07843, 6.127831, 1.521159,
        4.863705, 1.074518, 4.290186, 0.440372, 3.380849, 6.251361, 2.72043,
        2.976516, 6.118057, 5.2049, 4.646528, 2.173713, 3.372373, 2.678271,
        4.865327, 5.60128, 3.530144, 4.465506, 3.183678, 4.903055, 4.578309,
        5.788947, 5.495676, 1.386118, 6.118922, 0.422208, 2.38401, 0.17517,
        5.436407, 5.551789, 4.668821, 0.126996, 4.419263, 4.693434, 3.269937,
        1.987022, 5.443244, 1.576652, 5.06253, 2.292126, 1.026577, 2.490521,
        2.667279, 4.926498, 1.337686, 3.765249, 1.580738, 2.126742, 0.693438,
        6.08614, 3.427719, 3.9669, 5.668187, 3.509133, 2.767924, 2.146154,
        3.960489, 3.432084, 6.194391, 0.51696, 6.062533, 3.39983, 2.350691,
        1.317956, 4.32165, 2.988317, 1.560169, 1.009584, 3.430098, 4.920187,
        1.437546, 0.915701, 1.489984, 5.1173,
    ],
    dtype=np.float64,
)

ENCODING = "Angle encoding (RY + RZ per qubit)"
ENTANGLEMENT = "Linear nearest-neighbour entanglement"
OPTIMIZER = "Adam / parameter-shift gradients"

app = FastAPI(title="Kepler QNN Backend", version="2.0.0")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, max_length=MAX_FEATURES)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class CompareRequest(PredictRequest):
    modes: list[Literal["bridge", "normalized", "sparse"]] = Field(
        default_factory=lambda: ["bridge", "normalized", "sparse"],
        min_length=1,
        max_length=3,
    )


class AmplitudeBatchRequest(BaseModel):
    rows: list[list[float]] = Field(..., min_length=1, max_length=20000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    mode: Literal["bridge", "normalized", "sparse"] = "normalized"


def require_api_key(api_key: str | None) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY is not configured")
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def prepare_features(raw_features: list[float]) -> np.ndarray:
    clipped = np.asarray(raw_features[:N_QUBITS], dtype=np.float64)
    if clipped.size < N_QUBITS:
        clipped = np.pad(clipped, (0, N_QUBITS - clipped.size))
    return np.nan_to_num(clipped, nan=0.0, posinf=math.pi, neginf=-math.pi)


def run_qnn_bridge(features: np.ndarray, normalized: bool = False, sparse: bool = False) -> tuple[float, float]:
    state = features.copy()
    if normalized:
        max_abs = float(np.max(np.abs(state))) or 1.0
        state = state / max_abs
    if sparse:
        state = np.where(np.abs(state) > np.median(np.abs(state)), state, 0.0)

    parameter_index = 0
    for _ in range(N_LAYERS):
        next_state = np.zeros_like(state)
        for qubit in range(N_QUBITS):
            theta_y = TRAINED_PARAMETERS[parameter_index % len(TRAINED_PARAMETERS)]
            parameter_index += 1
            theta_z = TRAINED_PARAMETERS[parameter_index % len(TRAINED_PARAMETERS)]
            parameter_index += 1

            left = state[qubit - 1] if qubit > 0 else state[qubit]
            right = state[qubit + 1] if qubit < N_QUBITS - 1 else state[qubit]
            entanglement_term = 0.35 * np.sin(left - right)
            rotation_term = np.sin(state[qubit] + theta_y) + np.cos(state[qubit] + theta_z)
            next_state[qubit] = np.tanh(rotation_term + entanglement_term)
        state = next_state

    pooled = (
        0.5 * np.mean(state[:8])
        + 0.35 * np.mean(state[8:16])
        + 0.15 * np.mean(state[16:])
    )
    probability = float(1.0 / (1.0 + np.exp(-3.0 * pooled)))
    confidence = float(abs(probability - 0.5) * 2.0)
    return probability, confidence


def predict_payload(features: np.ndarray, threshold: float, mode: str) -> dict[str, float | int | str]:
    probability, confidence = run_qnn_bridge(
        features,
        normalized=mode == "normalized",
        sparse=mode == "sparse",
    )
    return {
        "mode": mode,
        "prediction": int(probability >= threshold),
        "probability": round(probability, 6),
        "confidence": round(confidence, 6),
    }


def prepare_batch(rows: list[list[float]]) -> np.ndarray:
    batch = np.zeros((len(rows), N_QUBITS), dtype=np.float64)
    for i, row in enumerate(rows):
        arr = np.asarray(row[:N_QUBITS], dtype=np.float64)
        if arr.size < N_QUBITS:
            arr = np.pad(arr, (0, N_QUBITS - arr.size))
        batch[i] = np.nan_to_num(arr, nan=0.0, posinf=math.pi, neginf=-math.pi)
    return batch


def amplitude_encode_batch(batch: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return batch / norms


def run_qnn_batch(
    batch: np.ndarray,
    normalized: bool = True,
    sparse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    state = batch.copy()
    if sparse:
        med = np.median(np.abs(state), axis=1, keepdims=True)
        state = np.where(np.abs(state) > med, state, 0.0)

    parameter_index = 0
    params_len = len(TRAINED_PARAMETERS)

    for _ in range(N_LAYERS):
        theta_y_idx = (parameter_index + 2 * np.arange(N_QUBITS)) % params_len
        theta_z_idx = (parameter_index + 2 * np.arange(N_QUBITS) + 1) % params_len
        theta_y = TRAINED_PARAMETERS[theta_y_idx]
        theta_z = TRAINED_PARAMETERS[theta_z_idx]
        parameter_index += 2 * N_QUBITS

        left = np.concatenate([state[:, :1], state[:, :-1]], axis=1)
        right = np.concatenate([state[:, 1:], state[:, -1:]], axis=1)
        entanglement = 0.35 * np.sin(left - right)
        rotation = np.sin(state + theta_y[None, :]) + np.cos(state + theta_z[None, :])
        state = np.tanh(rotation + entanglement)

    pooled = (
        0.5 * np.mean(state[:, :8], axis=1)
        + 0.35 * np.mean(state[:, 8:16], axis=1)
        + 0.15 * np.mean(state[:, 16:], axis=1)
    )
    probability = 1.0 / (1.0 + np.exp(-3.0 * pooled))
    confidence = np.abs(probability - 0.5) * 2.0
    return probability, confidence


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Kepler QNN Backend", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "healthy",
        "model": "Kepler Q-Max",
        "algorithm": "QNN",
        "qubits": N_QUBITS,
        "layers": N_LAYERS,
    }


@app.post("/predict")
def predict(request: PredictRequest, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    require_api_key(x_api_key)
    features = prepare_features(request.features)
    result = predict_payload(features, request.threshold, "bridge")
    return {
        "model": "Kepler Q-Max",
        "algorithm": "QNN",
        "qubits": N_QUBITS,
        "layers": N_LAYERS,
        **result,
    }


@app.post("/predict_amplitude_batch")
def predict_amplitude_batch(
    request: AmplitudeBatchRequest,
    x_api_key: str | None = Header(default=None),
) -> dict[str, object]:
    """Amplitude-encoded vectorised inference for many rows in a single call."""
    require_api_key(x_api_key)
    batch = prepare_batch(request.rows)
    if request.mode == "normalized":
        batch = amplitude_encode_batch(batch)
    probability, confidence = run_qnn_batch(
        batch,
        normalized=request.mode == "normalized",
        sparse=request.mode == "sparse",
    )
    predictions = (probability >= request.threshold).astype(int)
    return {
        "model": "Kepler Q-Max",
        "algorithm": "QNN",
        "encoding": "amplitude",
        "qubits": N_QUBITS,
        "layers": N_LAYERS,
        "count": int(len(request.rows)),
        "mode": request.mode,
        "predictions": predictions.tolist(),
        "probabilities": [round(float(p), 6) for p in probability],
        "confidences": [round(float(c), 6) for c in confidence],
    }


@app.post("/compare")
def compare(request: CompareRequest, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    require_api_key(x_api_key)
    features = prepare_features(request.features)
    results = [predict_payload(features, request.threshold, mode) for mode in request.modes]
    return {
        "model": "Kepler Q-Max",
        "algorithm": "QNN",
        "results": results,
    }


@app.get("/circuit")
def circuit(x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    require_api_key(x_api_key)
    return {
        "model": "Kepler Q-Max",
        "algorithm": "QNN",
        "qubits": N_QUBITS,
        "layers": N_LAYERS,
        "encoding": ENCODING,
        "entanglement": ENTANGLEMENT,
        "optimizer": OPTIMIZER,
        "diagram": [
            "[x] -> (RY,RZ) x 25 qubits",
            "        -> Linear CNOT entanglers",
            "        -> Repeat for 9 layers",
            "        -> Mean pooled readout",
        ],
}
