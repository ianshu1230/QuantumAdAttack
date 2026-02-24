from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import pennylane as qml

from config import CFG
from utils import (
    seed_everything,
    make_ckpt_dir,
    load_ckpt,
    save_ckpt,
    eval_accuracy_search,
    write_csv,
)

from modules.vqc import VQC
from dataGen import make_loaders


def parse_args() -> CFG:
    p = argparse.ArgumentParser("Train VQC classifier (synthetic)")

    # ---- io ----
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default="vqc_synth")

    # ---- data gen ----
    p.add_argument("--dataset", type=str, default="two_moons", choices=["two_moons"])
    p.add_argument("--n_samples", type=int, default=800)
    p.add_argument("--noise", type=float, default=0.12)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--standardize", action="store_true")

    # ---- training ----
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--batch_log", action="store_true")
    p.add_argument("--seed", type=int, default=87)

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda")

    # ---- VQC ----
    p.add_argument("--encoder", type=str, default="angle_ry")
    p.add_argument("--n_qubits", type=int, default=2)      #  two_moons is 2D, start from 2
    p.add_argument("--vqc_layers", type=int, default=2)
    p.add_argument("--hadamard", action="store_true")

    # ---- resume ----
    p.add_argument("--resume_path", type=str, default="")

    a = p.parse_args()

    dev = a.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    return CFG(
        outdir=a.outdir,
        exp_name=a.exp_name,

        dataset=a.dataset,
        n_samples=a.n_samples,
        noise=a.noise,
        random_state=a.random_state,
        test_ratio=a.test_ratio,
        batch_size=a.batch_size,
        standardize=a.standardize,

        epochs=a.epochs,
        lr=a.lr,
        gamma=a.gamma,
        batch_log=a.batch_log,
        seed=a.seed,

        device=dev,

        num_classes=-1,   # infer later
        in_dim=-1,        # infer later

        encoder=a.encoder,
        n_qubits=a.n_qubits,
        vqc_layers=a.vqc_layers,
        hadamard=a.hadamard,

        resume_path=a.resume_path,
    )


# -------------------------
# Model
# -------------------------
class QuantumClassifier(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        self.vqc = VQC(cfg)
        #self.head = nn.Linear(self.vqc.n_qubits, cfg.num_classes)

        print(f"[model] encoder={cfg.encoder}, n_qubits={self.vqc.n_qubits}, vqc_layers={cfg.vqc_layers}")
        print(f"[model] in_dim={cfg.in_dim}, num_classes={cfg.num_classes}")

    def _preproc(self, x: torch.Tensor) -> torch.Tensor:
        """
        For synthetic datasets, x is already (B, D). Keep it simple.
        """
        if x.dim() != 2:
            raise ValueError(f"Expect x as (B,D), got {tuple(x.shape)}")
        if x.size(1) < self.cfg.n_qubits and "amplitude" not in self.cfg.encoder.lower():
            raise ValueError(f"x dim {x.size(1)} < n_qubits {self.cfg.n_qubits} for angle encoder.")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._preproc(x)
        qfeats = self.vqc(feats)         # (B, n_qubits)
        logits = self.head(qfeats)       # (B, num_classes)
        return logits


def draw_circuit_once(model: QuantumClassifier, example_x: torch.Tensor, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    qml.drawer.use_style("black_white")

    with torch.no_grad():
        feats = model._preproc(example_x[:1]).detach()

    fig, ax = qml.draw_mpl(model.vqc.circuit)(feats, model.vqc.theta)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Circuit diagram saved to: {out_png}")
    

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    root_dir = Path(cfg.outdir)

    # q2_L2
    base_dir = root_dir / cfg.dataset / cfg.encoder / f"q{cfg.n_qubits}_L{cfg.vqc_layers}"

    # vqc_h / vqc
    vqc_tag = "vqc_h" if cfg.hadamard else "vqc"

    outdir = base_dir / vqc_tag
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader = make_loaders(cfg)
    print(f"[data] dataset={cfg.dataset}, in_dim={cfg.in_dim}, num_classes={cfg.num_classes}")

    # sanity: for angle encoders, ensure n_qubits <= in_dim
    if "amplitude" not in cfg.encoder.lower() and cfg.n_qubits > cfg.in_dim:
        raise ValueError(f"angle encoder needs n_qubits <= in_dim, got n_qubits={cfg.n_qubits}, in_dim={cfg.in_dim}")

    # Model
    model = QuantumClassifier(cfg).to(cfg.device)

    # Warmup forward
    x0, y0 = next(iter(train_loader))
    x0 = x0.to(cfg.device)
    _ = model(x0)

    # Draw circuit
    draw_circuit_once(model, x0, outdir / "circuit.png")

    # Optim
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[params] Trainable: {trainable_params}, Total: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    start_epoch = 1
    global_iter = 0

    if cfg.resume_path:
        rp = Path(cfg.resume_path)
        if rp.exists():
            start_epoch, best_acc = load_ckpt(rp, model, optimizer, cfg.device)
            print(f"[resume] {rp} | start_epoch={start_epoch} | best_acc={best_acc:.4f}")
        else:
            print(f"[resume] resume_path not found: {rp}")

    log_rows: List[List[Any]] = [["epoch", "iter", "train_loss", "test_acc", "trainable_params", "total_params"]]

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()

        for x, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            global_iter += 1
            log_rows.append([epoch, global_iter, float(loss.item()), "", "", ""])

            if cfg.batch_log:
                print(f"Epoch {epoch:02d} | Iter {global_iter:06d} | Loss {loss.item():.4f}")

        model.eval()
        test_acc = eval_accuracy_search(model, test_loader, cfg.device)

        log_rows.append([epoch, global_iter, "", float(test_acc), trainable_params, total_params])
        print(f"[epoch {epoch:02d}] Test Acc = {test_acc:.4f}")

        write_csv(log_rows, outdir / "loss.csv")
        save_ckpt(ckpt_dir / "last_search.pth", model, optimizer, epoch, best_acc, cfg)

        if test_acc > best_acc:
            best_acc = test_acc
            save_ckpt(ckpt_dir / "best_search.pth", model, optimizer, epoch, best_acc, cfg)
            print(f"[ckpt] Saved BEST: acc={best_acc:.4f}")

        scheduler.step()

    print("[done] Best acc:", best_acc)
    print("[done] Saved to:", str(outdir))


if __name__ == "__main__":
    main()