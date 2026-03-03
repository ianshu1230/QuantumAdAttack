import os
import random
import numpy as np
from pathlib import Path
import csv
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from config import CFG
from dataclasses import asdict


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def build_mnist_loaders(cfg):
    tfm = T.Compose([T.ToTensor()])
    root = str(Path(cfg.outdir).parent / "data")
    
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test_set  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    
    return train_loader, test_loader, cfg.num_classes
    

def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerows(rows)


def make_ckpt_dir(outdir, vqc, cfg):
    path = outdir / vqc /f"{cfg.reducer}_{cfg.encoder}"
    if cfg.encoder == "reupload":
        path = path / f"{cfg.reupload_layers}"
    return (path / "checkpoints")


def save_ckpt(path, model, optimizer, epoch, best_acc, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_acc": float(best_acc),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": asdict(cfg),
        },
        path,
    )


def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_acc = float(ckpt.get("best_acc", 0.0))
    return start_epoch, best_acc


@torch.no_grad()
def eval_accuracy_search(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        # Handle ensemble models that return (logits, logits_all, weights)
        logits = output[0] if isinstance(output, tuple) else output
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
