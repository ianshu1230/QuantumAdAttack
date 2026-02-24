# config.py
from dataclasses import dataclass

@dataclass
class CFG:
    # io
    outdir: str
    exp_name: str

    # training
    epochs: int
    lr: float
    gamma: float
    batch_log: bool
    seed: int

    # device
    device: str

    # data generation (NEW)
    dataset: str          # e.g., "two_moons"
    n_samples: int
    noise: float
    random_state: int
    test_ratio: float
    batch_size: int
    standardize: bool

    # labels / dims
    num_classes: int      # will be inferred from y if set to -1
    in_dim: int           # will be inferred from X if set to -1

    # VQC (match your VQC.py)
    encoder: str
    n_qubits: int
    vqc_layers: int
    hadamard: bool

    # resume
    resume_path: str