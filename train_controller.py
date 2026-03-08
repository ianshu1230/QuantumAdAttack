from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import pennylane as qml

from modules.vqc import EnsembleSharedVQC
from dataGen import make_loaders


# =========================================================
# IO utils
# =========================================================
def write_csv(rows: List[List], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_model_only(path: Path, model: nn.Module, device: str) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return ckpt


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Controller-only adversarial training for ensemble shared-VQC")

    # ---- io ----
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default="ensemble_controller_adv")
    p.add_argument("--pretrained_path", type=str, required=True)

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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--controller_lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=87)
    p.add_argument("--batch_log", action="store_true")

    # ---- loss weights ----
    p.add_argument("--lambda_clean", type=float, default=1.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_gate", type=float, default=0.1)

    # ---- train attack ----
    p.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "pgd"])
    p.add_argument("--attack_eps", type=float, default=0.15)
    p.add_argument("--attack_alpha", type=float, default=0.02)
    p.add_argument("--attack_steps", type=int, default=7)
    p.add_argument("--attack_random_start", action="store_true")

    # ---- eval attack ----
    p.add_argument("--eval_attack", type=str, default="pgd", choices=["fgsm", "pgd"])
    p.add_argument("--eval_eps", type=float, default=0.15)
    p.add_argument("--eval_alpha", type=float, default=0.02)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--eval_random_start", action="store_true")

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda")

    # ---- VQC ----
    p.add_argument("--encoder", type=str, default="ensemble")
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--vqc_layers", type=int, default=2)
    p.add_argument("--hadamard", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)

    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    return args


# =========================================================
# Model
# =========================================================
class QuantumClassifier(nn.Module):
    def __init__(self, cfg):
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
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() != 2:
            raise ValueError(f"Expect x as (B,D) or (B,C,H,W), got {tuple(x.shape)}")
        return x

    def forward_quantum(
        self,
        feats: torch.Tensor,
        use_controller: bool = True,
        return_controller: bool = False,
        return_branch_outputs: bool = False,
    ):
        """
        直接在 training script 內手動組 VQC forward，
        這樣不需要改你 modules/vqc.py 的 forward 介面。
        """
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)

        B = feats.size(0)
        ctrl_features = feats

        q_features = feats
        if q_features.size(1) > self.vqc.n_qubits:
            q_features = q_features[:, :self.vqc.n_qubits]

        # branch outputs
        branch_outputs = []
        for enc in self.vqc.encoder_list:
            out_enc = self.vqc._run_single_encoder_batch(q_features, enc)   # (B, Q)
            branch_outputs.append(out_enc)
        branch_outputs = torch.stack(branch_outputs, dim=1)  # (B, E, Q)

        # controller weights
        base_logits = self.vqc.alpha.unsqueeze(0).expand(B, -1)  # (B, E)

        if use_controller:
            if not hasattr(self.vqc, "controller"):
                raise ValueError("Residual controller not found: model.vqc.controller")
            delta_logits = self.vqc.controller(ctrl_features)     # (B, E)
            mixed_logits = base_logits + delta_logits
        else:
            delta_logits = torch.zeros_like(base_logits)
            mixed_logits = base_logits

        tau = getattr(self.vqc, "temperature", getattr(self.cfg, "temperature", 1.0))
        arch_w = torch.softmax(mixed_logits / tau, dim=-1)       # (B, E)

        qfeats = torch.einsum("be,beq->bq", arch_w, branch_outputs)
        qfeats = qfeats.to(dtype=torch.float32, device=feats.device)

        outputs = [qfeats]
        if return_branch_outputs:
            outputs.append(branch_outputs)
        if return_controller:
            outputs.extend([arch_w, mixed_logits, delta_logits])

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def forward(
        self,
        x: torch.Tensor,
        use_controller: bool = True,
        return_controller: bool = False,
        return_branch_outputs: bool = False,
    ):
        feats = self._preproc(x)

        q_out = self.forward_quantum(
            feats,
            use_controller=use_controller,
            return_controller=return_controller,
            return_branch_outputs=return_branch_outputs,
        )

        if isinstance(q_out, tuple):
            qfeats = q_out[0]
            logits = self.head(qfeats)
            return (logits,) + q_out[1:]

        logits = self.head(q_out)
        return logits


# =========================================================
# Visualization
# =========================================================
def draw_circuit_once(model: QuantumClassifier, example_x: torch.Tensor, out_png: Path) -> None:
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


# =========================================================
# Freeze / Unfreeze
# =========================================================
def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_controller_only(model: QuantumClassifier) -> None:
    freeze_all_params(model)

    if not hasattr(model.vqc, "controller"):
        raise ValueError("model.vqc has no attribute 'controller'")

    for p in model.vqc.controller.parameters():
        p.requires_grad = True


# =========================================================
# Attack
# =========================================================
def get_clip_bounds(x: torch.Tensor, dataset: str):
    # MNIST-like image => clamp [0, 1]
    if dataset.lower() == "mnist":
        return 0.0, 1.0
    # two_moons / standardized features => no clip
    return None, None


def fgsm_attack(
    model: QuantumClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    dataset: str,
    use_controller: bool,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    clip_min, clip_max = get_clip_bounds(x, dataset)

    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv, use_controller=use_controller)
    loss = F.cross_entropy(logits, y)

    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    x_adv = x_adv + eps * grad.sign()

    if clip_min is not None and clip_max is not None:
        x_adv = x_adv.clamp(clip_min, clip_max)

    x_adv = x_adv.detach()

    if was_training:
        model.train()
    return x_adv


def pgd_attack(
    model: QuantumClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    dataset: str,
    use_controller: bool,
    random_start: bool = False,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    clip_min, clip_max = get_clip_bounds(x, dataset)

    x_orig = x.detach()
    x_adv = x_orig.clone()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        if clip_min is not None and clip_max is not None:
            x_adv = x_adv.clamp(clip_min, clip_max)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv, use_controller=use_controller)
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = x_orig + delta
            if clip_min is not None and clip_max is not None:
                x_adv = x_adv.clamp(clip_min, clip_max)

    x_adv = x_adv.detach()

    if was_training:
        model.train()
    return x_adv


def make_attack(
    model: QuantumClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    dataset: str,
    use_controller: bool,
    attack_name: str,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
) -> torch.Tensor:
    if attack_name == "fgsm":
        return fgsm_attack(
            model=model,
            x=x,
            y=y,
            eps=eps,
            dataset=dataset,
            use_controller=use_controller,
        )
    elif attack_name == "pgd":
        return pgd_attack(
            model=model,
            x=x,
            y=y,
            eps=eps,
            alpha=alpha,
            steps=steps,
            dataset=dataset,
            use_controller=use_controller,
            random_start=random_start,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def eval_clean(
    model: QuantumClassifier,
    loader,
    device: str,
    use_controller: bool,
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, use_controller=use_controller)
        loss = F.cross_entropy(logits, y, reduction="sum")

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss


def eval_adv(
    model: QuantumClassifier,
    loader,
    device: str,
    dataset: str,
    use_controller: bool,
    attack_name: str,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in tqdm.tqdm(loader, desc=f"EvalAdv(ctrl={use_controller})", leave=False):
        x = x.to(device)
        y = y.to(device)

        x_adv = make_attack(
            model=model,
            x=x,
            y=y,
            dataset=dataset,
            use_controller=use_controller,
            attack_name=attack_name,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )

        with torch.no_grad():
            logits = model(x_adv, use_controller=use_controller)
            loss = F.cross_entropy(logits, y, reduction="sum")
            pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss


@torch.no_grad()
def collect_gate_stats_clean(
    model: QuantumClassifier,
    loader,
    device: str,
) -> List[float]:
    model.eval()
    gate_sum = None
    total = 0

    for x, _ in loader:
        x = x.to(device)
        logits, arch_w, mixed_logits, delta_logits = model(
            x,
            use_controller=True,
            return_controller=True,
        )
        batch_gate = arch_w.sum(dim=0)  # (E,)
        gate_sum = batch_gate if gate_sum is None else gate_sum + batch_gate
        total += x.size(0)

    gate_mean = gate_sum / max(total, 1)
    return gate_mean.detach().cpu().tolist()


def collect_gate_stats_adv(
    model: QuantumClassifier,
    loader,
    device: str,
    dataset: str,
    attack_name: str,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
) -> List[float]:
    model.eval()
    gate_sum = None
    total = 0

    for x, y in tqdm.tqdm(loader, desc="GateAdv", leave=False):
        x = x.to(device)
        y = y.to(device)

        x_adv = make_attack(
            model=model,
            x=x,
            y=y,
            dataset=dataset,
            use_controller=True,
            attack_name=attack_name,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )

        with torch.no_grad():
            logits, arch_w, mixed_logits, delta_logits = model(
                x_adv,
                use_controller=True,
                return_controller=True,
            )
            batch_gate = arch_w.sum(dim=0)

        gate_sum = batch_gate if gate_sum is None else gate_sum + batch_gate
        total += x.size(0)

    gate_mean = gate_sum / max(total, 1)
    return gate_mean.detach().cpu().tolist()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    seed_everything(args.seed)

    # --------------------------------------------------
    # make cfg-like namespace for your existing make_loaders / model
    # --------------------------------------------------
    cfg = SimpleNamespace(**vars(args))
    cfg.num_classes = -1
    cfg.in_dim = -1

    # Data
    train_loader, test_loader = make_loaders(cfg)
    print(f"[data] dataset={cfg.dataset}, in_dim={cfg.in_dim}, num_classes={cfg.num_classes}")

    if cfg.n_qubits > cfg.in_dim:
        raise ValueError(
            f"Current ensemble uses angle-style encoders only, "
            f"so require n_qubits <= in_dim. "
            f"Got n_qubits={cfg.n_qubits}, in_dim={cfg.in_dim}"
        )

    # Output dir
    root_dir = Path(cfg.outdir)
    encoder_tag = "ensemble"
    base_dir = root_dir / cfg.dataset / encoder_tag / f"q{cfg.n_qubits}_L{cfg.vqc_layers}"
    vqc_tag = "vqc_h" if cfg.hadamard else "vqc"
    atk_tag = f"train_{cfg.attack}_eps{cfg.attack_eps}_a{cfg.attack_alpha}_s{cfg.attack_steps}"
    eval_tag = f"eval_{cfg.eval_attack}_eps{cfg.eval_eps}_a{cfg.eval_alpha}_s{cfg.eval_steps}"
    outdir = base_dir / vqc_tag / "controller_adv" / f"{atk_tag}__{eval_tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ctrl_dir = outdir / "controller_weights"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), outdir / "config.json")

    # Model
    model = QuantumClassifier(cfg).to(cfg.device)

    # Warmup forward
    x0, y0 = next(iter(train_loader))
    x0 = x0.to(cfg.device)
    _ = model(x0, use_controller=True)

    # Draw circuit
    draw_circuit_once(model, x0, outdir / "circuit.png")

    # Load pretrained
    pretrained_path = Path(cfg.pretrained_path)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"pretrained_path not found: {pretrained_path}")

    ckpt = load_model_only(pretrained_path, model, cfg.device)
    print(f"[pretrained] loaded from: {pretrained_path}")

    # Train only controller
    unfreeze_controller_only(model)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("[trainable params]")
    for n in trainable_names:
        print("  ", n)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[params] Trainable: {trainable_params}, Total: {total_params}")

    optimizer = optim.Adam(
        trainable,
        lr=cfg.controller_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # Logs
    loss_rows: List[List] = [[
        "epoch",
        "iter",
        "train_clean_loss",
        "train_adv_loss",
        "train_gate_loss",
        "train_total_loss",
        "train_clean_acc",
        "train_adv_acc",
        "clean_ctrl_acc",
        "clean_ctrl_loss",
        "adv_ctrl_acc",
        "adv_ctrl_loss",
        "clean_base_acc",
        "clean_base_loss",
        "adv_base_acc",
        "adv_base_loss",
        "trainable_params",
        "total_params",
    ]]

    encoder_rows: List[List] = [["epoch"] + list(model.vqc.encoder_list)]
    gate_clean_rows: List[List] = [["epoch"] + list(model.vqc.encoder_list)]
    gate_adv_rows: List[List] = [["epoch"] + list(model.vqc.encoder_list)]

    best_metric = -1.0
    global_iter = 0

    # --------------------------------------------------
    # training
    # --------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        epoch_clean_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_gate_loss = 0.0
        epoch_total_loss = 0.0
        epoch_clean_correct = 0
        epoch_adv_correct = 0
        epoch_total = 0

        for x, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            # generate adversarial examples against controller-enabled model
            x_adv = make_attack(
                model=model,
                x=x,
                y=y,
                dataset=cfg.dataset,
                use_controller=True,
                attack_name=cfg.attack,
                eps=cfg.attack_eps,
                alpha=cfg.attack_alpha,
                steps=cfg.attack_steps,
                random_start=cfg.attack_random_start,
            )

            optimizer.zero_grad(set_to_none=True)

            # clean forward
            logits_clean, gate_clean, mixed_clean, delta_clean = model(
                x,
                use_controller=True,
                return_controller=True,
            )

            # adv forward
            logits_adv, gate_adv, mixed_adv, delta_adv = model(
                x_adv,
                use_controller=True,
                return_controller=True,
            )

            loss_clean = F.cross_entropy(logits_clean, y)
            loss_adv = F.cross_entropy(logits_adv, y)

            if cfg.lambda_gate > 0:
                loss_gate = F.kl_div(
                    torch.log(gate_adv + 1e-8),
                    gate_clean.detach(),
                    reduction="batchmean",
                )
            else:
                loss_gate = torch.tensor(0.0, device=cfg.device)

            loss = (
                cfg.lambda_clean * loss_clean
                + cfg.lambda_adv * loss_adv
                + cfg.lambda_gate * loss_gate
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_clean = logits_clean.argmax(dim=1)
                pred_adv = logits_adv.argmax(dim=1)

                epoch_clean_correct += (pred_clean == y).sum().item()
                epoch_adv_correct += (pred_adv == y).sum().item()
                epoch_total += y.size(0)

                bs = y.size(0)
                epoch_clean_loss += loss_clean.item() * bs
                epoch_adv_loss += loss_adv.item() * bs
                epoch_gate_loss += loss_gate.item() * bs
                epoch_total_loss += loss.item() * bs

            global_iter += 1

            if cfg.batch_log:
                print(
                    f"Epoch {epoch:02d} | Iter {global_iter:06d} | "
                    f"clean={loss_clean.item():.4f} adv={loss_adv.item():.4f} "
                    f"gate={loss_gate.item():.4f} total={loss.item():.4f}"
                )

        scheduler.step()

        train_clean_loss = epoch_clean_loss / max(epoch_total, 1)
        train_adv_loss = epoch_adv_loss / max(epoch_total, 1)
        train_gate_loss = epoch_gate_loss / max(epoch_total, 1)
        train_total_loss = epoch_total_loss / max(epoch_total, 1)
        train_clean_acc = epoch_clean_correct / max(epoch_total, 1)
        train_adv_acc = epoch_adv_correct / max(epoch_total, 1)

        # --------------------------------------------------
        # evaluation: four cases
        # --------------------------------------------------
        clean_ctrl_acc, clean_ctrl_loss = eval_clean(
            model=model,
            loader=test_loader,
            device=cfg.device,
            use_controller=True,
        )

        adv_ctrl_acc, adv_ctrl_loss = eval_adv(
            model=model,
            loader=test_loader,
            device=cfg.device,
            dataset=cfg.dataset,
            use_controller=True,
            attack_name=cfg.eval_attack,
            eps=cfg.eval_eps,
            alpha=cfg.eval_alpha,
            steps=cfg.eval_steps,
            random_start=cfg.eval_random_start,
        )

        clean_base_acc, clean_base_loss = eval_clean(
            model=model,
            loader=test_loader,
            device=cfg.device,
            use_controller=False,
        )

        adv_base_acc, adv_base_loss = eval_adv(
            model=model,
            loader=test_loader,
            device=cfg.device,
            dataset=cfg.dataset,
            use_controller=False,
            attack_name=cfg.eval_attack,
            eps=cfg.eval_eps,
            alpha=cfg.eval_alpha,
            steps=cfg.eval_steps,
            random_start=cfg.eval_random_start,
        )

        print(
            f"[epoch {epoch:02d}] "
            f"train_clean_acc={train_clean_acc:.4f} "
            f"train_adv_acc={train_adv_acc:.4f} | "
            f"clean_ctrl={clean_ctrl_acc:.4f} "
            f"adv_ctrl={adv_ctrl_acc:.4f} | "
            f"clean_base={clean_base_acc:.4f} "
            f"adv_base={adv_base_acc:.4f}"
        )

        # --------------------------------------------------
        # save alpha weights
        # --------------------------------------------------
        with torch.no_grad():
            base_w = torch.softmax(model.vqc.alpha, dim=0).detach().cpu().tolist()
        encoder_rows.append([epoch] + [float(w) for w in base_w])
        write_csv(encoder_rows, outdir / "encoder_weights.csv")

        # --------------------------------------------------
        # save gate stats
        # --------------------------------------------------
        gate_clean_mean = collect_gate_stats_clean(
            model=model,
            loader=test_loader,
            device=cfg.device,
        )
        gate_adv_mean = collect_gate_stats_adv(
            model=model,
            loader=test_loader,
            device=cfg.device,
            dataset=cfg.dataset,
            attack_name=cfg.eval_attack,
            eps=cfg.eval_eps,
            alpha=cfg.eval_alpha,
            steps=cfg.eval_steps,
            random_start=cfg.eval_random_start,
        )

        gate_clean_rows.append([epoch] + [float(v) for v in gate_clean_mean])
        gate_adv_rows.append([epoch] + [float(v) for v in gate_adv_mean])
        write_csv(gate_clean_rows, outdir / "gate_stats_clean.csv")
        write_csv(gate_adv_rows, outdir / "gate_stats_adv.csv")

        # --------------------------------------------------
        # save per-epoch controller weights
        # --------------------------------------------------
        torch.save(model.vqc.controller.state_dict(), ctrl_dir / f"epoch_{epoch:03d}.pt")

        # --------------------------------------------------
        # save all metrics
        # --------------------------------------------------
        loss_rows.append([
            epoch,
            global_iter,
            float(train_clean_loss),
            float(train_adv_loss),
            float(train_gate_loss),
            float(train_total_loss),
            float(train_clean_acc),
            float(train_adv_acc),
            float(clean_ctrl_acc),
            float(clean_ctrl_loss),
            float(adv_ctrl_acc),
            float(adv_ctrl_loss),
            float(clean_base_acc),
            float(clean_base_loss),
            float(adv_base_acc),
            float(adv_base_loss),
            trainable_params,
            total_params,
        ])
        write_csv(loss_rows, outdir / "loss.csv")

        # --------------------------------------------------
        # save last
        # --------------------------------------------------
        save_ckpt(
            ckpt_dir / "last_search.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            args=args,
        )
        torch.save(model.vqc.controller.state_dict(), ckpt_dir / "last_controller.pt")

        # --------------------------------------------------
        # save best (用 adv_ctrl_acc 當主指標)
        # --------------------------------------------------
        if adv_ctrl_acc > best_metric:
            best_metric = adv_ctrl_acc
            save_ckpt(
                ckpt_dir / "best_search.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )
            torch.save(model.vqc.controller.state_dict(), ckpt_dir / "best_controller.pt")
            print(f"[ckpt] Saved BEST controller by adv_ctrl_acc={best_metric:.4f}")

    print(f"[done] best adv_ctrl_acc = {best_metric:.4f}")
    print(f"[done] saved to: {outdir}")


if __name__ == "__main__":
    main()