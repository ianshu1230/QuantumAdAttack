from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import pennylane as qml

from config import CFG
from utils import (
    seed_everything,
    load_ckpt,
    save_ckpt,
    eval_accuracy_search,
    write_csv,
)

from modules.vqc import EnsembleSharedVQC
from dataGen import make_loaders


def parse_args() -> CFG:
    p = argparse.ArgumentParser("Train ensemble shared-VQC classifier")

    # ---- io ----
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default="ensemble_vqc")

    # ---- data ----
    p.add_argument("--dataset", type=str, default="two_moons", choices=["two_moons", "mnist"])
    p.add_argument("--n_samples", type=int, default=800)
    p.add_argument("--noise", type=float, default=0.12)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--standardize", action="store_true")

    # ---- MNIST subset ----
    p.add_argument("--digits", type=str, default=None)
    p.add_argument("--data_root", type=str, default="./datasets")
    p.add_argument("--img_size", type=int, default=4)

    # ---- training ----
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--batch_log", action="store_true")
    p.add_argument("--seed", type=int, default=87)

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda")

    # ---- VQC ----
    # ensemble 模式下這個值只做紀錄 / 相容，實際固定視為 ensemble
    p.add_argument("--encoder", type=str, default="ensemble")
    p.add_argument("--n_qubits", type=int, default=2)
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

        data_root=a.data_root,
        img_size=a.img_size,

        device=dev,

        num_classes=-1,   # inferred in make_loaders
        in_dim=-1,        # inferred in make_loaders
        digits=a.digits,

        encoder="ensemble",
        n_qubits=a.n_qubits,
        vqc_layers=a.vqc_layers,
        hadamard=a.hadamard,

        resume_path=a.resume_path,
    )


class QuantumClassifier(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        self.vqc = EnsembleSharedVQC(cfg)
        self.head = nn.Linear(self.vqc.n_qubits, cfg.num_classes)

        candidates = getattr(self.vqc, "encoder_list", [])
        print(
            f"[model] encoder={cfg.encoder}, "
            f"candidates={candidates}, "
            f"n_qubits={self.vqc.n_qubits}, "
            f"vqc_layers={cfg.vqc_layers}"
        )
        print(f"[model] in_dim={cfg.in_dim}, num_classes={cfg.num_classes}")

    def _preproc(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, D) or (B, C, H, W)
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() != 2:
            raise ValueError(f"Expect x as (B,D) or (B,C,H,W), got {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._preproc(x)         # (B, D)
        qfeats = self.vqc(feats)         # (B, n_qubits)
        logits = self.head(qfeats)       # (B, num_classes)
        return logits


def draw_circuit_once(model: QuantumClassifier, example_x: torch.Tensor, out_png: Path) -> None:
    """
    畫一個代表性電路圖。
    你的 EnsembleSharedVQC 若有:
      - model.vqc.circuit
      - model.vqc.theta
    就會成功畫出。
    """
    if not hasattr(model.vqc, "circuit") or not hasattr(model.vqc, "theta"):
        print("[viz] Skip circuit drawing because model.vqc.circuit or theta is missing.")
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    qml.drawer.use_style("black_white")

    with torch.no_grad():
        feats = model._preproc(example_x[:1]).detach()

    try:
        fig, ax = qml.draw_mpl(model.vqc.circuit)(feats, model.vqc.theta)
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] Circuit diagram saved to: {out_png}")
    except Exception as e:
        print(f"[viz] Skip drawing circuit due to error: {e}")


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    root_dir = Path(cfg.outdir)

    # ensemble 固定命名
    encoder_tag = "ensemble"
    base_dir = root_dir / cfg.dataset / encoder_tag / f"q{cfg.n_qubits}_L{cfg.vqc_layers}"

    vqc_tag = "vqc_h" if cfg.hadamard else "vqc"
    outdir = base_dir / vqc_tag
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader = make_loaders(cfg)
    print(f"[data] dataset={cfg.dataset}, in_dim={cfg.in_dim}, num_classes={cfg.num_classes}")

    # 目前 ensemble 先不含 amplitude，因此要求 n_qubits <= in_dim
    if cfg.n_qubits > cfg.in_dim:
        raise ValueError(
            f"Current ensemble uses angle-style encoders only, "
            f"so require n_qubits <= in_dim. "
            f"Got n_qubits={cfg.n_qubits}, in_dim={cfg.in_dim}"
        )

    # Model
    model = QuantumClassifier(cfg).to(cfg.device)

    # Warmup forward
    x0, y0 = next(iter(train_loader))
    x0 = x0.to(cfg.device)
    _ = model(x0)

    # Draw circuit
    draw_circuit_once(model, x0, outdir / "circuit.png")

    # Optimizer
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

    # main logs
    log_rows: List[List[Any]] = [
        ["epoch", "iter", "train_loss", "test_acc", "trainable_params", "total_params"]
    ]

    # architecture weight logs
    if hasattr(model.vqc, "encoder_list"):
        arch_log_rows: List[List[Any]] = [["epoch"] + list(model.vqc.encoder_list)]
    else:
        arch_log_rows = [["epoch"]]

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

        # save encoder weights
        if hasattr(model.vqc, "alpha") and hasattr(model.vqc, "encoder_list"):
            with torch.no_grad():
                arch_w = torch.softmax(model.vqc.alpha, dim=0).detach().cpu().numpy()

            arch_log_rows.append([epoch] + [float(w) for w in arch_w.tolist()])
            write_csv(arch_log_rows, outdir / "encoder_weights.csv")

            print(
                f"[epoch {epoch:02d}] encoder_weights = " +
                ", ".join(f"{n}:{w:.4f}" for n, w in zip(model.vqc.encoder_list, arch_w))
            )

        write_csv(log_rows, outdir / "loss.csv")

        if test_acc > best_acc:
            best_acc = test_acc
            save_ckpt(ckpt_dir / "best_search.pth", model, optimizer, epoch, best_acc, cfg)
            print(f"[ckpt] Saved BEST: acc={best_acc:.4f}")

        save_ckpt(ckpt_dir / "last_search.pth", model, optimizer, epoch, best_acc, cfg)

        scheduler.step()

    print("[done] Best acc:", best_acc)
    print("[done] Saved to:", str(outdir))


if __name__ == "__main__":
    main()