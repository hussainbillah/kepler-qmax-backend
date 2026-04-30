"""
HYBRID DEPLOYMENT BRIDGE — Kepler Q-Max
Unified interface for Qiskit, PyTorch, and PennyLane models.
"""
import numpy as np
from typing import Dict, Any, Literal


class HybridModelBridge:
    SUPPORTED_BACKENDS = ["qiskit", "pytorch", "pennylane"]

    def __init__(self, backend: Literal["qiskit", "pytorch", "pennylane"] = "pennylane"):
        assert backend in self.SUPPORTED_BACKENDS
        self.backend_name = backend
        self.model = self._load_backend(backend)

    def _load_backend(self, backend: str):
        if backend == "qiskit":
            from trained_model_qiskit import TrainedQuantumModel
            return TrainedQuantumModel()
        elif backend == "pytorch":
            from trained_model_pytorch import load_trained_model
            return load_trained_model()
        elif backend == "pennylane":
            from trained_model_pennylane import PennyLaneQMLModel
            return PennyLaneQMLModel()

    def predict(self, x) -> Dict[str, Any]:
        x = np.array(x, dtype=np.float64)
        if self.backend_name == "qiskit":
            return self.model.predict(x)
        elif self.backend_name == "pytorch":
            import torch
            with torch.no_grad():
                tensor_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                if tensor_x.shape[1] < 25:
                    pad = torch.zeros(1, 25 - tensor_x.shape[1])
                    tensor_x = torch.cat([tensor_x, pad], dim=1)
                tensor_x = tensor_x[:, :25]
                output = self.model(tensor_x)
                pred = torch.argmax(output, dim=1).item()
                logits = output.tolist()[0]
                conf = float(torch.softmax(output, dim=1).max().item())
            return {"prediction": pred, "backend": "pytorch", "logits": logits, "confidence": conf}
        elif self.backend_name == "pennylane":
            return self.model.predict(x)

    def compare_backends(self, x) -> Dict[str, Any]:
        results = {}
        for backend in self.SUPPORTED_BACKENDS:
            try:
                bridge = HybridModelBridge(backend=backend)
                results[backend] = bridge.predict(x)
            except Exception as e:
                results[backend] = {"error": str(e)}
        return results
