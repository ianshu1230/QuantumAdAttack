import torch
import torch.nn as nn
import pennylane as qml
import math
from modules import encoders


class VQC(nn.Module):
    """
    Encoder -> H (optional) → [RX, RY, RZ, CNOT]^layers → measure Z
    """ 
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.vqc_layers
        self.output_dim = cfg.num_classes
        self.device = cfg.device
        self.encoder = cfg.encoder
        self.hadamard = cfg.hadamard
        self._initialized = False
        
        if "angle" in self.encoder:
            self.n_qubits = cfg.num_classes
        if "amplitude" in self.encoder:
            self.n_qubits = math.ceil(math.log2(cfg.num_classes))
    
    def _lazy_init(self):
        # Random initialization of parameters
        self.theta = nn.Parameter(0.01 * torch.randn(self.layers, self.n_qubits, 3, 
                                                     dtype=torch.float32, device=self.device))
        # Use lightning for faster computation
        # self.qdev = qml.device("default.qubit", wires=self.n_qubits)
        self.qdev = qml.device("lightning.qubit", wires=self.n_qubits)
        self._build_qnode()
        self._initialized = True
    
    def _apply_encoder(self, features):
        encoder = self.encoder.lower()
        n = self.n_qubits
        
        if encoder == "angle_rx":
            encoders.angle_rx_encoder(features, n)
        elif encoder == "angle_ry":
            encoders.angle_ry_encoder(features, n)
        elif encoder == "angle_rz":
            encoders.angle_rz_encoder(features, n)
        elif encoder == "amplitude":
            encoders.amplitude_encoder(features, n)
        elif encoder == "h_angle_rx":
            encoders.h_angle_rx_encoder(features, n)
        elif encoder == "h_angle_ry":
            encoders.h_angle_ry_encoder(features, n)
        elif encoder == "h_angle_rz":
            encoders.h_angle_rz_encoder(features, n)
        elif encoder == "h_amplitude":
            encoders.h_amplitude_encoder(features, n)
        else:
            raise ValueError(f"Unknown encoder {self.encoder}.")
    
    def _build_qnode(self):
        n = self.n_qubits
    
        @qml.qnode(self.qdev, interface="torch", diff_method="adjoint")
        def circuit(features, theta):
            self._apply_encoder(features)
            
            if self.hadamard:
                for q in range(n):
                    qml.Hadamard(wires=q)

            for l in range(self.layers):
                # Rotations
                for q in range(n):
                    qml.RX(theta[l, q, 0], wires=q)
                    qml.RY(theta[l, q, 1], wires=q)
                    qml.RZ(theta[l, q, 2], wires=q)
                # CNOT
                for q in range(0, n - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, n - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        if "amplitude" in self.encoder:
            circuit = qml.transforms.broadcast_expand(circuit)
        self.circuit = circuit
        
    def forward(self, features):
        if not self._initialized:
            self._lazy_init()
        
        # Handle different input shapes
        # if states.dim() == 3 and states.shape[-1] == 1:
        #     vec = states.flatten(start_dim=1)  # (B, 2^n)
        # elif states.dim() == 2:
        #     vec = states  # Already (B, 2^n)
        # elif states.dim() == 1:
        #     vec = states.unsqueeze(0)  # (1, 2^n)
        # else:
        #     raise ValueError(f"Unexpected states shape: {states.shape}")
        
        # Normalization
        # norm = torch.norm(cnn_features, dim=1, keepdim=True)
        # cnn_features_norm = cnn_features / (norm + 1e-8)
        # qfeats = self.circuit(cnn_features_norm, self.theta)

        qfeats = self.circuit(features, self.theta)
        qfeats = torch.stack(qfeats, dim=1)
        return qfeats.to(dtype=torch.float32, device=self.device)