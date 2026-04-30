# Kepler Q-Max Quantum Inference API

FastAPI server exposing the trained QAOA (p=10, 25 qubits) model via REST.

## Files
- `main.py` — FastAPI app (`/predict`, `/compare`, `/circuit`, `/health`)
- `hybrid_deployment_bridge.py` — Unified backend router
- `trained_model_pennylane.py` — PennyLane backend (default)
- `trained_model_pytorch.py` — PyTorch hybrid classifier
- `trained_model_qiskit.py` — Qiskit Aer simulator backend
- `requirements.txt`, `Dockerfile`, `render.yaml`

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input":[0.1,0.2,0.3], "backend":"pennylane"}'
```

## Deploy to Render (recommended — free tier works)
1. Push this folder to a GitHub repo.
2. Go to https://render.com → **New → Web Service** → connect repo.
3. Render auto-detects `render.yaml`. Click **Apply**.
4. After build, copy the URL: `https://kepler-qmax-api.onrender.com`.
5. Render auto-generates `KEPLER_API_KEY` — copy it from **Environment** tab.

## Deploy to Railway
1. Push to GitHub.
2. https://railway.app → New Project → Deploy from GitHub.
3. Add env var `KEPLER_API_KEY = <your-secret>`.

## Deploy to HuggingFace Spaces (free GPU optional)
1. Create Space → SDK: **Docker**.
2. Upload these files. Spaces auto-builds from `Dockerfile`.

## Connect to Kepler Q-Max (Lovable)
After deploy, give me:
- **API URL** (e.g. `https://kepler-qmax-api.onrender.com`)
- **API Key** (the `KEPLER_API_KEY` value)

I will store both as Lovable Cloud secrets and wire the "Start Training"
button to call `/predict` with the user's uploaded dataset.

## Endpoints

### POST /predict
```json
{ "input": [0.1, 0.2, ...], "backend": "pennylane" }
```
Returns: `{ prediction, confidence, expectation_values, backend }`

### POST /compare
Runs all 3 backends on the same input.

### GET /circuit
Returns ASCII drawing of the quantum circuit + metadata.

### GET /health
For uptime monitoring (Render uses this).
