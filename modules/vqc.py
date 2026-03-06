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
        
        self.n_qubits = cfg.n_qubits
        # if "angle" in self.encoder:
        #     self.n_qubits = cfg.num_classes
        # if "amplitude" in self.encoder:
        #     self.n_qubits = math.ceil(math.log2(cfg.num_classes))
    
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

class EnsembleSharedVQC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.vqc_layers
        self.output_dim = cfg.num_classes
        self.device = cfg.device
        self.n_qubits = cfg.n_qubits
        self.hadamard = getattr(cfg, "hadamard", False)

        # 候選 encoder
        self.encoder_list = [
            "angle_rx",
            "angle_ry",
            "h_angle_rx",
            "h_angle_ry",
            #"amplitude",
        ]
        self.n_encoders = len(self.encoder_list)

        # shared VQC parameters: 所有 encoder 共用
        self.theta = nn.Parameter(
            0.01 * torch.randn(
                self.layers, self.n_qubits, 3,
                dtype=torch.float32
            )
        )

        # structural weights / architecture weights
        # 經 softmax 後變成每個 encoder 的權重
        self.alpha = nn.Parameter(torch.zeros(self.n_encoders, dtype=torch.float32))

        # quantum device
        self.qdev = qml.device("lightning.qubit", wires=self.n_qubits)

        # 建立多個 qnode，每個 qnode 只差在 encoder
        self.qnodes = nn.ModuleList()  # 只是佔位，不放 qnode
        self._build_qnodes()

    def _encode_input(self, features, encoder_name):
        """
        這個函式只負責在 qnode 裡呼叫對應 encoder。
        """
        n = self.n_qubits
        enc = encoder_name.lower()

        if enc == "angle_rx":
            encoders.angle_rx_encoder(features, n)
        elif enc == "angle_ry":
            encoders.angle_ry_encoder(features, n)
        elif enc == "angle_rz":
            encoders.angle_rz_encoder(features, n)
        elif enc == "amplitude":
            encoders.amplitude_encoder(features, n)
        elif enc == "h_angle_rx":
            encoders.h_angle_rx_encoder(features, n)
        elif enc == "h_angle_ry":
            encoders.h_angle_ry_encoder(features, n)
        elif enc == "h_angle_rz":
            encoders.h_angle_rz_encoder(features, n)
        elif enc == "h_amplitude":
            encoders.h_amplitude_encoder(features, n)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def _variational_block(self, theta):
        n = self.n_qubits

        if self.hadamard:
            for q in range(n):
                qml.Hadamard(wires=q)

        for l in range(self.layers):
            for q in range(n):
                qml.RX(theta[l, q, 0], wires=q)
                qml.RY(theta[l, q, 1], wires=q)
                qml.RZ(theta[l, q, 2], wires=q)

            # brick entanglement
            for q in range(0, n - 1, 2):
                qml.CNOT(wires=[q, q + 1])
            for q in range(1, n - 1, 2):
                qml.CNOT(wires=[q, q + 1])

    def _build_single_qnode(self, encoder_name):
        @qml.qnode(self.qdev, interface="torch", diff_method="adjoint")
        def circuit(features, theta):
            self._encode_input(features, encoder_name)
            self._variational_block(theta)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def _build_qnodes(self):
        self.circuits = {}
        for enc in self.encoder_list:
            circ = self._build_single_qnode(enc)
            # amplitude 常常需要 broadcast_expand
            if "amplitude" in enc:
                circ = qml.transforms.broadcast_expand(circ)
            self.circuits[enc] = circ

    def _run_single_encoder_batch(self, x, encoder_name):
        """
        x: (B, D)
        return: (B, n_qubits)
        """
        outs = []
        for i in range(x.shape[0]):
            qi = self.circuits[encoder_name](x[i], self.theta)
            qi = torch.stack(qi) if isinstance(qi, (list, tuple)) else qi
            outs.append(qi)
        return torch.stack(outs, dim=0)

    def forward(self, features, return_branch_outputs=False):
        """
        features: (B, D) 或單筆 (D,)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # structural weights
        arch_w = torch.softmax(self.alpha, dim=0)   # (n_encoders,)

        branch_outputs = []
        for enc in self.encoder_list:
            out_enc = self._run_single_encoder_batch(features, enc)  # (B, n_qubits)
            branch_outputs.append(out_enc)

        # (E, B, Q)
        branch_outputs = torch.stack(branch_outputs, dim=0)

        # weighted sum over encoders
        y = torch.einsum("e,ebq->bq", arch_w, branch_outputs)

        if return_branch_outputs:
            return y, arch_w, branch_outputs
        return y