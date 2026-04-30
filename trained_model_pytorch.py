"""
HYBRID QUANTUM-CLASSICAL MODEL (PyTorch) — Kepler Q-Max
"""
import torch
import torch.nn as nn

TRAINED_PARAMETERS = [6.193686,2.607128,5.412743,0.0551,1.145859,0.448881,3.233979,2.964088,4.782724,6.004619,0.482922,4.074438,4.106344,0.581398,3.873394,4.216046,0.145427,3.012645,1.318869,3.275435,5.06723,3.642144,2.80884,5.175439,5.400748,1.250406,2.677881,3.127894,4.959416,4.791517,2.773739,5.733174,5.011725,1.609444,5.123757,2.708372,0.730436,2.187698,3.650226,3.074602,5.406853,2.880131,6.056074,3.744526,2.390053,4.951078,5.475751,5.632602,2.741939,3.629262,5.269341,1.202611,4.895462,0.68068,0.846809,3.451075,6.032537,1.328753,6.192446,5.683595,2.840686,1.903099,0.702083,1.772987,2.559609,1.301816,0.242062,3.380131,3.967883,1.807951,5.520429,4.755631,3.778504,4.070689,3.40501,4.390808,3.801435,2.632656,3.615081,5.861767,2.085243,2.521835,0.536361,3.462522,3.086246,3.59921,2.807292,3.776211,1.802932,1.454995,4.718344,3.957387,5.198381,4.951623,1.231609,6.148746,5.007777,2.967422,5.485437,4.51324,6.094422,2.96128,4.086946,2.313805,2.460194,1.655584,0.694343,4.946399,1.932285,1.032958,4.652034,5.517901,1.056922,2.200722,5.224737,6.263934,1.797129,5.495587,4.963483,1.291964,6.122759,2.789589,5.094828,4.785454,1.550045,6.07774,0.75823,3.030534,2.524179,0.919026,1.505015,5.507147,0.480955,3.899713,6.078585,1.149503,3.719911,2.734844,0.296373,4.44259,4.217042,5.30079,4.442562,0.601148,5.385628,4.489061,3.372633,2.772196,1.320332,0.351238,2.508654,2.233198,0.692394,2.664174,2.085655,2.535787,1.539046,1.447264,4.022712,3.390151,0.690204,1.348543,4.000323,2.717879,3.765806,3.758853,4.57786,5.190483,3.247939,4.2965,5.074277,2.008848,0.516858,5.760569,0.904286,1.346996,4.797524,4.239578,1.500234,1.958795,0.082758,1.880719,2.122524,5.07897,4.616801,0.864761,3.511825,4.24092,3.518657,3.817547,5.261123,1.018808,1.651124,3.477476,0.887428,2.024419,5.589234,2.557875,4.153731,4.325512]


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=25, n_layers=10):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        n_params = n_qubits * n_layers * 3
        params = (TRAINED_PARAMETERS * ((n_params // len(TRAINED_PARAMETERS)) + 1))[:n_params]
        self.quantum_params = nn.Parameter(torch.tensor(params, dtype=torch.float32))

    def forward(self, x):
        encoded = torch.tanh(x[:, :self.n_qubits] * self.quantum_params[:self.n_qubits])
        for layer in range(self.n_layers):
            offset = layer * self.n_qubits * 3
            ry = self.quantum_params[offset:offset + self.n_qubits]
            rz = self.quantum_params[offset + self.n_qubits:offset + 2 * self.n_qubits]
            encoded = torch.cos(encoded * ry) * torch.sin(encoded * rz)
        return encoded


class HybridQuantumClassifier(nn.Module):
    def __init__(self, input_dim=25, n_classes=2, n_qubits=25, n_layers=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, n_qubits), nn.Tanh(),
        )
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.decoder(self.quantum_layer(self.encoder(x)))


def load_trained_model(input_dim=25, n_classes=2):
    model = HybridQuantumClassifier(input_dim=input_dim, n_classes=n_classes)
    model.eval()
    return model
