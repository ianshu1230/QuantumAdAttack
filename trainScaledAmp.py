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

from modules.vqc import VQC, AmplitudeScalingGlobal
from dataGen import make_loaders


# =========================================================
# IO utils
# =========================================================
def write_csv(rows: List[List], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def seed_everything(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer,
              epoch: int, best_metric: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, path)


def load_model_only(path: Path, model: nn.Module, device: str) -> Dict:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[load] missing keys:")
    for k in missing:
        print("  ", k)
    print("[load] unexpected keys:")
    for k in unexpected:
        print("  ", k)
    return ckpt


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("AmplitudeScaling adversarial training for VQC")

    # ---- io ----
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default="scaled_amp_adv")
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to pretrained VQC checkpoint. None = train from scratch.")

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
    p.add_argument("--scaler_lr", type=float, default=1e-3)
    p.add_argument("--vqc_lr", type=float, default=None,
                   help="LR for VQC theta + head in full mode. Defaults to scaler_lr.")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=87)
    p.add_argument("--batch_log", action="store_true")

    # ---- loss weights ----
    p.add_argument("--lambda_clean", type=float, default=1.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)

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

    # ---- train mode ----
    p.add_argument(
        "--train_mode",
        type=str,
        default="scaler_only",
        choices=["scaler_only", "full"],
        help=(
            "scaler_only: freeze VQC theta + head, only train AmplitudeScaler. "
            "full: train scaler + VQC theta + head end-to-end."
        ),
    )

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda")

    # ---- VQC ----
    p.add_argument("--encoder", type=str, default="amplitude",
                   choices=["amplitude", "h_amplitude"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--vqc_layers", type=int, default=2)
    p.add_argument("--hadamard", action="store_true")

    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    return args


# =========================================================
# Model
# =========================================================
class ScaledAmplitudeClassifier(nn.Module):
    """
    AmplitudeScalingGlobal (input-dependent) → VQC (amplitude encoder) → Linear head.

    The scaler learns to re-weight the amplitude vector per sample before
    encoding. PennyLane's AmplitudeEmbedding normalizes internally, so the
    scaler affects relative amplitude ratios, not absolute magnitude.
    """
    def __init__(self, cfg):
        super().__init__()
        self.scaler = AmplitudeScalingGlobal(n_qubits=cfg.n_qubits, in_dim=cfg.in_dim)
        self.vqc = VQC(cfg)
        self.head = nn.Linear(cfg.n_qubits, cfg.num_classes)

        print(
            f"[model] encoder={cfg.encoder}, n_qubits={cfg.n_qubits}, "
            f"vqc_layers={cfg.vqc_layers}, in_dim={cfg.in_dim}, num_classes={cfg.num_classes}"
        )

    def _preproc(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() != 2:
            raise ValueError(f"Expect (B,D) or (B,C,H,W), got {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor, use_scaler: bool = True) -> torch.Tensor:
        x = self._preproc(x)
        if use_scaler:
            x = self.scaler(x)       # (B, D), input-dependent scaling
        q_out = self.vqc(x)          # (B, n_qubits)
        return self.head(q_out)      # (B, num_classes)


# =========================================================
# Freeze / Unfreeze
# =========================================================
def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_scaler_only(model: ScaledAmplitudeClassifier) -> None:
    freeze_all_params(model)
    for p in model.scaler.parameters():
        p.requires_grad = True


def unfreeze_all_params(model: ScaledAmplitudeClassifier) -> None:
    for p in model.parameters():
        p.requires_grad = True


# =========================================================
# Attack
# =========================================================
def get_clip_bounds(dataset: str):
    if dataset.lower() == "mnist":
        return 0.0, 1.0
    return None, None


def fgsm_attack(model, x, y, eps, dataset, use_scaler) -> torch.Tensor:
    was_training = model.training
    model.eval()
    clip_min, clip_max = get_clip_bounds(dataset)

    x_adv = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv, use_scaler=use_scaler), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + eps * grad.sign()

    if clip_min is not None:
        x_adv = x_adv.clamp(clip_min, clip_max)

    x_adv = x_adv.detach()
    if was_training:
        model.train()
    return x_adv


def pgd_attack(model, x, y, eps, alpha, steps, dataset, use_scaler,
               random_start: bool = False) -> torch.Tensor:
    was_training = model.training
    model.eval()
    clip_min, clip_max = get_clip_bounds(dataset)

    x_orig = x.detach()
    x_adv = x_orig.clone()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        if clip_min is not None:
            x_adv = x_adv.clamp(clip_min, clip_max)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv, use_scaler=use_scaler), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = x_orig + delta
            if clip_min is not None:
                x_adv = x_adv.clamp(clip_min, clip_max)

    x_adv = x_adv.detach()
    if was_training:
        model.train()
    return x_adv


def make_attack(model, x, y, dataset, use_scaler,
                attack_name, eps, alpha, steps, random_start) -> torch.Tensor:
    if attack_name == "fgsm":
        return fgsm_attack(model, x, y, eps, dataset, use_scaler)
    elif attack_name == "pgd":
        return pgd_attack(model, x, y, eps, alpha, steps, dataset, use_scaler, random_start)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def eval_clean(model, loader, device, use_scaler) -> Tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, use_scaler=use_scaler)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1), loss_sum / max(total, 1)


def eval_adv(model, loader, device, dataset, use_scaler,
             attack_name, eps, alpha, steps, random_start) -> Tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in tqdm.tqdm(loader, desc=f"EvalAdv(scaler={use_scaler})", leave=False):
        x, y = x.to(device), y.to(device)
        x_adv = make_attack(model, x, y, dataset, use_scaler,
                            attack_name, eps, alpha, steps, random_start)
        with torch.no_grad():
            logits = model(x_adv, use_scaler=use_scaler)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1), loss_sum / max(total, 1)


@torch.no_grad()
def collect_scale_stats(model: ScaledAmplitudeClassifier, loader, device) -> float:
    """Mean scale value (softplus output) across the test set."""
    model.eval()
    scale_sum = 0.0
    total = 0
    for x, _ in loader:
        x = x.to(device)
        if x.dim() == 4:
            x = x.flatten(1)
        scale = F.softplus(model.scaler.linear(x))  # (B, 1)
        scale_sum += scale.sum().item()
        total += x.size(0)
    return scale_sum / max(total, 1)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg = SimpleNamespace(**vars(args))
    cfg.num_classes = -1
    cfg.in_dim = -1

    # Data
    train_loader, test_loader = make_loaders(cfg)

    x_probe, _ = next(iter(train_loader))
    effective_in_dim = x_probe.flatten(1).shape[1] if x_probe.dim() == 4 else x_probe.shape[1]
    cfg.in_dim = effective_in_dim
    print(f"[data] dataset={cfg.dataset}, in_dim={cfg.in_dim}, num_classes={cfg.num_classes}")

    # Output dir
    # runs/{dataset}/scaled_amp/{encoder}/q{n}_L{layers}/{vqc_tag}/{mode_dir}/{atk_tag}__{eval_tag}/
    root_dir = Path(cfg.outdir)
    base_dir = (root_dir / cfg.dataset / "scaled_amp" / cfg.encoder
                / f"q{cfg.n_qubits}_L{cfg.vqc_layers}")
    vqc_tag  = "vqc_h" if cfg.hadamard else "vqc"
    atk_tag  = f"train_{cfg.attack}_eps{cfg.attack_eps}_a{cfg.attack_alpha}_s{cfg.attack_steps}"
    eval_tag = f"eval_{cfg.eval_attack}_eps{cfg.eval_eps}_a{cfg.eval_alpha}_s{cfg.eval_steps}"
    mode_dir = "scaler_adv" if cfg.train_mode == "scaler_only" else "full_adv"
    outdir   = base_dir / vqc_tag / mode_dir / f"{atk_tag}__{eval_tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir   = outdir / "checkpoints"
    scaler_dir = outdir / "scaler_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), outdir / "config.json")

    # Model
    model = ScaledAmplitudeClassifier(cfg).to(cfg.device)

    # Warmup forward (lazy-init VQC qnode)
    x0, _ = next(iter(train_loader))
    _ = model(x0.to(cfg.device), use_scaler=True)

    # Load pretrained (optional)
    if cfg.pretrained_path is not None:
        pretrained_path = Path(cfg.pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"pretrained_path not found: {pretrained_path}")
        _ = load_model_only(pretrained_path, model, cfg.device)
        print(f"[pretrained] loaded from: {pretrained_path}")
    else:
        print("[pretrained] training from scratch")

    # Baseline eval
    clean_scaler_acc0, _ = eval_clean(model, test_loader, cfg.device, use_scaler=True)
    clean_base_acc0,   _ = eval_clean(model, test_loader, cfg.device, use_scaler=False)
    print(
        f"[before training] "
        f"clean_scaler_acc={clean_scaler_acc0:.4f} | clean_base_acc={clean_base_acc0:.4f}"
    )

    # Freeze / unfreeze
    if cfg.train_mode == "scaler_only":
        print("[train_mode] scaler_only: freeze VQC theta + head, train scaler only")
        unfreeze_scaler_only(model)
    else:
        print("[train_mode] full: train scaler + VQC theta + head end-to-end")
        unfreeze_all_params(model)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("[trainable params]")
    for n in trainable_names:
        print("  ", n)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"[params] Trainable: {trainable_params}, Total: {total_params}")

    # Optimizer (separate lr groups for full mode)
    if cfg.train_mode == "scaler_only":
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.scaler_lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        vqc_lr = cfg.vqc_lr if cfg.vqc_lr is not None else cfg.scaler_lr
        scaler_params    = list(model.scaler.parameters())
        scaler_param_ids = {id(p) for p in scaler_params}
        vqc_params       = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in scaler_param_ids
        ]
        optimizer = optim.Adam(
            [
                {"params": scaler_params, "lr": cfg.scaler_lr},
                {"params": vqc_params,    "lr": vqc_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
        print(
            f"[optimizer] scaler_lr={cfg.scaler_lr}, vqc_lr={vqc_lr}, "
            f"scaler_params={sum(p.numel() for p in scaler_params)}, "
            f"vqc_params={sum(p.numel() for p in vqc_params)}"
        )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # Logs
    loss_rows: List[List] = [[
        "epoch", "iter",
        "train_clean_loss", "train_adv_loss", "train_total_loss",
        "train_clean_acc",  "train_adv_acc",
        "clean_scaler_acc", "clean_scaler_loss",
        "adv_scaler_acc",   "adv_scaler_loss",
        "clean_base_acc",   "clean_base_loss",
        "adv_base_acc",     "adv_base_loss",
        "mean_scale",
        "trainable_params", "total_params",
    ]]
    scale_rows: List[List] = [["epoch", "mean_scale"]]

    best_metric = -1.0
    global_iter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        epoch_clean_loss = epoch_adv_loss = epoch_total_loss = 0.0
        epoch_clean_correct = epoch_adv_correct = epoch_total = 0

        for x, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            x, y = x.to(cfg.device), y.to(cfg.device)

            # Generate adversarial examples with scaler active
            x_adv = make_attack(
                model=model, x=x, y=y, dataset=cfg.dataset,
                use_scaler=True,
                attack_name=cfg.attack,
                eps=cfg.attack_eps, alpha=cfg.attack_alpha,
                steps=cfg.attack_steps, random_start=cfg.attack_random_start,
            )

            optimizer.zero_grad(set_to_none=True)

            logits_clean = model(x,     use_scaler=True)
            logits_adv   = model(x_adv, use_scaler=True)

            loss_clean = F.cross_entropy(logits_clean, y)
            loss_adv   = F.cross_entropy(logits_adv,   y)
            loss = cfg.lambda_clean * loss_clean + cfg.lambda_adv * loss_adv

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                bs = y.size(0)
                epoch_clean_correct += (logits_clean.argmax(1) == y).sum().item()
                epoch_adv_correct   += (logits_adv.argmax(1)   == y).sum().item()
                epoch_total         += bs
                epoch_clean_loss    += loss_clean.item() * bs
                epoch_adv_loss      += loss_adv.item()   * bs
                epoch_total_loss    += loss.item()        * bs

            global_iter += 1

            if cfg.batch_log:
                print(
                    f"Epoch {epoch:02d} | Iter {global_iter:06d} | "
                    f"clean={loss_clean.item():.4f} adv={loss_adv.item():.4f} "
                    f"total={loss.item():.4f}"
                )

        scheduler.step()

        n = max(epoch_total, 1)
        train_clean_loss = epoch_clean_loss / n
        train_adv_loss   = epoch_adv_loss   / n
        train_total_loss = epoch_total_loss / n
        train_clean_acc  = epoch_clean_correct / n
        train_adv_acc    = epoch_adv_correct   / n

        # Eval with scaler
        clean_scaler_acc, clean_scaler_loss = eval_clean(
            model, test_loader, cfg.device, use_scaler=True)
        adv_scaler_acc, adv_scaler_loss = eval_adv(
            model, test_loader, cfg.device, cfg.dataset, True,
            cfg.eval_attack, cfg.eval_eps, cfg.eval_alpha,
            cfg.eval_steps, cfg.eval_random_start)

        # Eval without scaler (pure VQC baseline)
        clean_base_acc, clean_base_loss = eval_clean(
            model, test_loader, cfg.device, use_scaler=False)
        adv_base_acc, adv_base_loss = eval_adv(
            model, test_loader, cfg.device, cfg.dataset, False,
            cfg.eval_attack, cfg.eval_eps, cfg.eval_alpha,
            cfg.eval_steps, cfg.eval_random_start)

        # Track mean scale
        mean_scale = collect_scale_stats(model, test_loader, cfg.device)
        scale_rows.append([epoch, float(mean_scale)])
        write_csv(scale_rows, outdir / "scale_stats.csv")

        print(
            f"[epoch {epoch:02d}] "
            f"train_clean={train_clean_acc:.4f} train_adv={train_adv_acc:.4f} | "
            f"clean_scaler={clean_scaler_acc:.4f} adv_scaler={adv_scaler_acc:.4f} | "
            f"clean_base={clean_base_acc:.4f} adv_base={adv_base_acc:.4f} | "
            f"mean_scale={mean_scale:.4f}"
        )

        loss_rows.append([
            epoch, global_iter,
            float(train_clean_loss), float(train_adv_loss), float(train_total_loss),
            float(train_clean_acc),  float(train_adv_acc),
            float(clean_scaler_acc), float(clean_scaler_loss),
            float(adv_scaler_acc),   float(adv_scaler_loss),
            float(clean_base_acc),   float(clean_base_loss),
            float(adv_base_acc),     float(adv_base_loss),
            float(mean_scale),
            trainable_params, total_params,
        ])
        write_csv(loss_rows, outdir / "loss.csv")

        # Save scaler weights every epoch
        torch.save(model.scaler.state_dict(), scaler_dir / f"epoch_{epoch:03d}.pt")

        # Save last checkpoint
        save_ckpt(ckpt_dir / "last.pth", model, optimizer, epoch, best_metric, args)
        torch.save(model.scaler.state_dict(), ckpt_dir / "last_scaler.pt")

        # Save best checkpoint (by adv accuracy with scaler)
        if adv_scaler_acc > best_metric:
            best_metric = adv_scaler_acc
            save_ckpt(ckpt_dir / "best.pth", model, optimizer, epoch, best_metric, args)
            torch.save(model.scaler.state_dict(), ckpt_dir / "best_scaler.pt")
            print(f"[ckpt] Saved BEST by adv_scaler_acc={best_metric:.4f}")

    print(f"[done] best adv_scaler_acc = {best_metric:.4f}")
    print(f"[done] saved to: {outdir}")


if __name__ == "__main__":
    main()
