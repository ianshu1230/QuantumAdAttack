import pennylane as qml
import math

def angle_rx_encoder(x, n_qubits: int):
    for i in range(n_qubits):
        qml.RX(math.pi * x[..., i], wires=i)


def angle_ry_encoder(x, n_qubits: int):
    for i in range(n_qubits):
        qml.RY(math.pi * x[..., i], wires=i)


def angle_rz_encoder(x, n_qubits: int):
    for i in range(n_qubits):
        qml.RZ(math.pi * x[..., i], wires=i)


def amplitude_encoder(x, n_qubits: int):
    dim = 2 ** n_qubits
    qml.AmplitudeEmbedding(x[..., :dim], wires=range(n_qubits), normalize=True, pad_with=0.0)


def h_angle_rx_encoder(x, n_qubits: int):
    """Hadamard + RX rotation."""
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for i in range(n_qubits):
        qml.RX(math.pi * x[..., i], wires=i)


def h_angle_ry_encoder(x, n_qubits: int):
    """Hadamard + RY rotation."""
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for i in range(n_qubits):
        qml.RY(math.pi * x[..., i], wires=i)


def h_angle_rz_encoder(x, n_qubits: int):
    """Hadamard + RZ rotation."""
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for i in range(n_qubits):
        qml.RZ(math.pi * x[..., i], wires=i)


def h_amplitude_encoder(x, n_qubits: int):
    """Hadamard + Amplitude encoding."""
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    dim = 2 ** n_qubits
    qml.AmplitudeEmbedding(x[..., :dim], wires=range(n_qubits), normalize=True, pad_with=0.0)
