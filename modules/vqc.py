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


class ResidualEncoderController(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_encoders: int,
        residual_scale: float = 0.5,
    ):
        super().__init__()

        self.residual_scale = residual_scale
        self.linear = nn.Linear(in_dim, n_encoders)

        nn.init.xavier_uniform_(self.linear.weight, gain=0.3)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        raw = self.linear(x)                    # (B, E)
        delta = self.residual_scale * torch.tanh(raw)
        return delta
class EnsembleSharedVQC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.vqc_layers
        self.output_dim = cfg.num_classes
        self.device = cfg.device
        self.n_qubits = cfg.n_qubits
        self.hadamard = getattr(cfg, "hadamard", False)

        self.encoder_list = [
            "angle_rx",
            "angle_ry",
            "h_angle_rx",
            "h_angle_ry",
        ]
        self.n_encoders = len(self.encoder_list)

        self.theta = nn.Parameter(
            0.01 * torch.randn(
                self.layers, self.n_qubits, 3,
                dtype=torch.float32
            )
        )

        self.controller = ResidualEncoderController(
            in_dim=self.n_qubits,
            n_encoders=self.n_encoders,
            residual_scale=getattr(cfg, "residual_scale", 0.5),
        )

        self.temperature = getattr(cfg, "temperature", 1.0)

        self.alpha = nn.Parameter(
            torch.zeros(self.n_encoders, dtype=torch.float32)
        )

        self.qdev = qml.device("lightning.qubit", wires=self.n_qubits)
        self._build_qnodes()

        self.circuit = self.circuits[self.encoder_list[0]]

    def _encode_input(self, features, encoder_name):
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
            if "amplitude" in enc:
                circ = qml.transforms.broadcast_expand(circ)
            self.circuits[enc] = circ

    def _run_single_encoder_batch(self, x, encoder_name):
        outs = []
        target_device = x.device

        for i in range(x.shape[0]):
            qi = self.circuits[encoder_name](x[i], self.theta)
            qi = torch.stack(qi) if isinstance(qi, (list, tuple)) else qi
            qi = qi.to(device=target_device, dtype=torch.float32)
            outs.append(qi)

        return torch.stack(outs, dim=0).to(device=target_device, dtype=torch.float32)

    def forward(self, features, use_controller=True, return_branch_outputs=False, return_controller=False):
        if features.dim() == 1:
            features = features.unsqueeze(0)

        B = features.size(0)

        ctrl_features = features
        if ctrl_features.size(1) > self.n_qubits:
            ctrl_features = ctrl_features[:, :self.n_qubits]

        q_features = features
        if q_features.size(1) > self.n_qubits:
            q_features = q_features[:, :self.n_qubits]

        if use_controller:
            delta_logits = self.controller(ctrl_features)
            base_logits = self.alpha.unsqueeze(0).expand(B, -1)
            mixed_logits = base_logits + delta_logits
        else:
            base_logits = self.alpha.unsqueeze(0).expand(B, -1)
            delta_logits = torch.zeros_like(base_logits)
            mixed_logits = base_logits

        arch_w = torch.softmax(mixed_logits / self.temperature, dim=-1)

        branch_outputs = []
        for enc in self.encoder_list:
            out_enc = self._run_single_encoder_batch(q_features, enc)
            out_enc = out_enc.to(device=features.device, dtype=torch.float32)
            branch_outputs.append(out_enc)

        branch_outputs = torch.stack(branch_outputs, dim=1).to(
            device=features.device, dtype=torch.float32
        )  # (B, E, Q)

        y = torch.einsum("be,beq->bq", arch_w, branch_outputs)
        y = y.to(dtype=torch.float32, device=features.device)

        outputs = [y]

        if return_branch_outputs:
            outputs.append(branch_outputs)

        if return_controller:
            outputs.extend([arch_w, mixed_logits, delta_logits])

        return outputs[0] if len(outputs) == 1 else tuple(outputs)
# class EnsembleSharedVQC(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.layers = cfg.vqc_layers
#         self.output_dim = cfg.num_classes
#         self.device = cfg.device
#         self.n_qubits = cfg.n_qubits
#         self.hadamard = getattr(cfg, "hadamard", False)

#         # 候選 encoder
#         self.encoder_list = [
#             "angle_rx",
#             "angle_ry",
#             "h_angle_rx",
#             "h_angle_ry",
#         ]
#         self.n_encoders = len(self.encoder_list)

#         # shared VQC parameters: 所有 encoder 共用
#         self.theta = nn.Parameter(
#             0.01 * torch.randn(
#                 self.layers, self.n_qubits, 3,
#                 dtype=torch.float32
#             )
#         )
#         self.controller = ResidualEncoderController(
#             in_dim=self.n_qubits,   # 通常是原始輸入維度
#             n_encoders=self.n_encoders,
#             #hidden_dim=getattr(cfg, "controller_hidden_dim", 64),
#             #num_layers=getattr(cfg, "controller_num_layers", 2),
#             #dropout=getattr(cfg, "controller_dropout", 0.0),
#             #use_layernorm=getattr(cfg, "controller_layernorm", False),
#             residual_scale=getattr(cfg, "residual_scale", 0.5),
#         )
#         self.temperature = getattr(cfg, "temperature", 1.0)
#         # architecture weights
#         self.alpha = nn.Parameter(
#             torch.zeros(self.n_encoders, dtype=torch.float32)
#         )

#         # quantum device
#         self.qdev = qml.device("lightning.qubit", wires=self.n_qubits)

#         # 建立多個 qnode
#         self._build_qnodes()

#         # 給 draw_circuit_once 用
#         # 固定拿第一個 encoder 當代表圖
#         self.circuit = self.circuits[self.encoder_list[0]]

#     def _encode_input(self, features, encoder_name):
#         n = self.n_qubits
#         enc = encoder_name.lower()

#         if enc == "angle_rx":
#             encoders.angle_rx_encoder(features, n)
#         elif enc == "angle_ry":
#             encoders.angle_ry_encoder(features, n)
#         elif enc == "angle_rz":
#             encoders.angle_rz_encoder(features, n)
#         elif enc == "amplitude":
#             encoders.amplitude_encoder(features, n)
#         elif enc == "h_angle_rx":
#             encoders.h_angle_rx_encoder(features, n)
#         elif enc == "h_angle_ry":
#             encoders.h_angle_ry_encoder(features, n)
#         elif enc == "h_angle_rz":
#             encoders.h_angle_rz_encoder(features, n)
#         elif enc == "h_amplitude":
#             encoders.h_amplitude_encoder(features, n)
#         else:
#             raise ValueError(f"Unknown encoder: {encoder_name}")

#     def _variational_block(self, theta):
#         n = self.n_qubits

#         if self.hadamard:
#             for q in range(n):
#                 qml.Hadamard(wires=q)

#         for l in range(self.layers):
#             for q in range(n):
#                 qml.RX(theta[l, q, 0], wires=q)
#                 qml.RY(theta[l, q, 1], wires=q)
#                 qml.RZ(theta[l, q, 2], wires=q)

#             for q in range(0, n - 1, 2):
#                 qml.CNOT(wires=[q, q + 1])
#             for q in range(1, n - 1, 2):
#                 qml.CNOT(wires=[q, q + 1])

#     def _build_single_qnode(self, encoder_name):
#         @qml.qnode(self.qdev, interface="torch", diff_method="adjoint")
#         def circuit(features, theta):
#             self._encode_input(features, encoder_name)
#             self._variational_block(theta)
#             return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
#         return circuit

#     def _build_qnodes(self):
#         self.circuits = {}
#         for enc in self.encoder_list:
#             circ = self._build_single_qnode(enc)
#             if "amplitude" in enc:
#                 circ = qml.transforms.broadcast_expand(circ)
#             self.circuits[enc] = circ

#     def _run_single_encoder_batch(self, x, encoder_name):
#         """
#         x: (B, D)
#         return: (B, n_qubits)
#         """
#         outs = []
#         target_device = x.device

#         for i in range(x.shape[0]):
#             qi = self.circuits[encoder_name](x[i], self.theta)
#             qi = torch.stack(qi) if isinstance(qi, (list, tuple)) else qi
#             qi = qi.to(device=target_device, dtype=torch.float32)
#             outs.append(qi)

#         return torch.stack(outs, dim=0).to(device=target_device, dtype=torch.float32)

#     def forward(self, features, return_branch_outputs=False, return_controller=False):
#         """
#         features: (B, D) or (D,)
#         """
#         if features.dim() == 1:
#             features = features.unsqueeze(0)

#         B = features.size(0)

#         # controller 看完整輸入
#         ctrl_features = features

#         # quantum encoder 若是 angle-family，只吃前 n_qubits 維
#         q_features = features
#         if q_features.size(1) > self.n_qubits:
#             q_features = q_features[:, :self.n_qubits]

#         # ----- residual controller -----
#         delta_logits = self.controller(ctrl_features)              # (B, E)
#         base_logits = self.alpha.unsqueeze(0).expand(B, -1)       # (B, E)
#         mixed_logits = base_logits + delta_logits                  # (B, E)
#         arch_w = torch.softmax(mixed_logits / self.temperature, dim=-1)  # (B, E)

#         # ----- run all branches -----
#         branch_outputs = []
#         for enc in self.vqc.encoder_list:
#             out_enc = self.vqc._run_single_encoder_batch(q_features, enc)
#             out_enc = out_enc.to(device=feats.device, dtype=torch.float32)
#             branch_outputs.append(out_enc)

#         branch_outputs = torch.stack(branch_outputs, dim=1).to(device=feats.device, dtype=torch.float32)

#         # weighted sum over encoders
#         y = torch.einsum("be,beq->bq", arch_w, branch_outputs)
#         y = y.to(dtype=torch.float32, device=features.device)

#         outputs = [y]

#         if return_branch_outputs:
#             outputs.append(branch_outputs)

#         if return_controller:
#             outputs.extend([arch_w, mixed_logits, delta_logits])

#         return outputs[0] if len(outputs) == 1 else tuple(outputs)