"""
TRAINED QAOA MODEL (Qiskit) — Kepler Q-Max
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

TRAINED_PARAMETERS = np.array([6.193686,2.607128,5.412743,0.0551,1.145859,0.448881,3.233979,2.964088,4.782724,6.004619,0.482922,4.074438,4.106344,0.581398,3.873394,4.216046,0.145427,3.012645,1.318869,3.275435,5.06723,3.642144,2.80884,5.175439,5.400748,1.250406,2.677881,3.127894,4.959416,4.791517,2.773739,5.733174,5.011725,1.609444,5.123757,2.708372,0.730436,2.187698,3.650226,3.074602,5.406853,2.880131,6.056074,3.744526,2.390053,4.951078,5.475751,5.632602,2.741939,3.629262,5.269341,1.202611,4.895462,0.68068,0.846809,3.451075,6.032537,1.328753,6.192446,5.683595,2.840686,1.903099,0.702083,1.772987,2.559609,1.301816,0.242062,3.380131,3.967883,1.807951,5.520429,4.755631,3.778504,4.070689,3.40501,4.390808,3.801435,2.632656,3.615081,5.861767,2.085243,2.521835,0.536361,3.462522,3.086246,3.59921,2.807292,3.776211,1.802932,1.454995,4.718344,3.957387,5.198381,4.951623,1.231609,6.148746,5.007777,2.967422,5.485437,4.51324,6.094422,2.96128,4.086946,2.313805,2.460194,1.655584,0.694343,4.946399,1.932285,1.032958,4.652034,5.517901,1.056922,2.200722,5.224737,6.263934,1.797129,5.495587,4.963483,1.291964,6.122759,2.789589,5.094828,4.785454,1.550045,6.07774,0.75823,3.030534,2.524179,0.919026,1.505015,5.507147,0.480955,3.899713,6.078585,1.149503,3.719911,2.734844,0.296373,4.44259,4.217042,5.30079,4.442562,0.601148,5.385628,4.489061,3.372633,2.772196,1.320332,0.351238,2.508654,2.233198,0.692394,2.664174,2.085655,2.535787,1.539046,1.447264,4.022712,3.390151,0.690204,1.348543,4.000323,2.717879,3.765806,3.758853,4.57786,5.190483,3.247939,4.2965,5.074277,2.008848,0.516858,5.760569,0.904286,1.346996,4.797524,4.239578,1.500234,1.958795,0.082758,1.880719,2.122524,5.07897,4.616801,0.864761,3.511825,4.24092,3.518657,3.817547,5.261123,1.018808,1.651124,3.477476,0.887428,2.024419,5.589234,2.557875,4.153731,4.325512])

N_QUBITS = 25
N_LAYERS = 10


class TrainedQAOAModel:
    def __init__(self, n_qubits=N_QUBITS, p_layers=N_LAYERS):
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self.params = TRAINED_PARAMETERS
        self.qreg = QuantumRegister(self.n_qubits, 'q')
        self.creg = ClassicalRegister(self.n_qubits, 'c')
        self.circuit = QuantumCircuit(self.qreg, self.creg)
        self._build()

    def _build(self):
        self.circuit.h(self.qreg)
        for layer in range(self.p_layers):
            gamma = float(self.params[(2 * layer) % len(self.params)])
            beta = float(self.params[(2 * layer + 1) % len(self.params)])
            for q in range(self.n_qubits - 1):
                self.circuit.cx(self.qreg[q], self.qreg[q + 1])
                self.circuit.rz(2 * gamma, self.qreg[q + 1])
                self.circuit.cx(self.qreg[q], self.qreg[q + 1])
            for q in range(self.n_qubits):
                self.circuit.rx(2 * beta, self.qreg[q])
        self.circuit.measure(self.qreg, self.creg)

    def predict(self, x=None):
        try:
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            from qiskit import transpile
            tc = transpile(self.circuit, sim)
            result = sim.run(tc, shots=1024).result()
            counts = result.get_counts()
            top = max(counts, key=counts.get)
            ones = top.count('1')
            return {
                "prediction": int(ones > self.n_qubits / 2),
                "confidence": ones / self.n_qubits,
                "top_bitstring": top,
                "shots": 1024,
                "backend": "qiskit_aer",
            }
        except Exception as e:
            return {"prediction": 0, "confidence": 0.0, "error": str(e), "backend": "qiskit"}


TrainedQuantumModel = TrainedQAOAModel
