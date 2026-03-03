# data_gen.py
import numpy as np
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
    make_gaussian_quantiles,
    make_hastie_10_2,
    make_swiss_roll,
    make_s_curve,
)
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import torch
from config import CFG

# >>> ADD THESE IMPORTS
import torchvision
import torchvision.transforms as T


# -------------------------
# Basic datasets
# -------------------------

def two_moons(n_samples=800, noise=0.12, random_state=42):
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

def circles(n_samples=800, noise=0.06, factor=0.5, random_state=42):
    return make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)

def blobs(n_samples=800, centers=3, cluster_std=1.2, random_state=42):
    return make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)

def classification_2d(n_samples=800, class_sep=1.0, flip_y=0.03, random_state=42):
    return make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

def gaussian_quantiles(n_samples=800, n_classes=3, random_state=42):
    return make_gaussian_quantiles(
        n_samples=n_samples, n_features=2, n_classes=n_classes, random_state=random_state
    )

def hastie_10_2(n_samples=800, random_state=42, use_first2=True):
    X, y = make_hastie_10_2(n_samples=n_samples, random_state=random_state)
    y = (y > 0).astype(np.int64)
    if use_first2:
        X = X[:, :2]
    return X, y


# -------------------------
# Manifold datasets
# -------------------------

def swiss_roll(n_samples=800, noise=0.25, random_state=42, pca2=True):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    if pca2:
        X = PCA(n_components=2, random_state=random_state).fit_transform(X)
    return X, t

def s_curve(n_samples=800, noise=0.25, random_state=42, pca2=True):
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)
    if pca2:
        X = PCA(n_components=2, random_state=random_state).fit_transform(X)
    return X, t


# ------------------------- 
# MNIST subset
# -------------------------
class Wrapped(torch.utils.data.Dataset):
    def __init__(self, base_ds, label_map):
        self.base_ds = base_ds
        self.label_map = label_map
        # precompute indices to keep
        self.idx = [i for i, (_, y) in enumerate(base_ds) if int(y) in label_map]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.base_ds[self.idx[i]]    # x: (1,28,28) float in [0,1]
        y = self.label_map[int(y)]          # remap to 0..K-1
        return x, torch.tensor(y, dtype=torch.long)

def make_mnist_subset(cfg):
    """
    Returns:
      train_loader, test_loader, K
    where K = number of selected digits (i.e., num_classes after remap)
    """
    # transforms
    tf = T.Compose([
    T.Resize((cfg.img_size, cfg.img_size)),  # e.g. 14 or 7
    T.ToTensor(),
    ])

    # download: set cfg.mnist_download = True if you want auto-download
    download = bool(getattr(cfg, "mnist_download", True))

    train = torchvision.datasets.MNIST(root=cfg.data_root, train=True,  download=download, transform=tf)
    test  = torchvision.datasets.MNIST(root=cfg.data_root, train=False, download=download, transform=tf)
    # digits selection (e.g., cfg.digits = [0,1,7] or "0,1,7")
    digits = getattr(cfg, "digits", None)
    if digits is None:
        # default: use all 10 digits
        digits = list(range(10))
    elif isinstance(digits, str):
        # allow "0,1,7" or "0 1 7"
        sep = "," if "," in digits else None
        digits = [int(x) for x in digits.replace(" ", "," if sep is None else " ").replace(" ", ",").split(",") if x != ""]
    else:
        digits = [int(d) for d in digits]

    digits = list(sorted(set(digits)))
    if len(digits) == 0:
        raise ValueError("cfg.digits is empty. Example: cfg.digits=[0,1]")

    label_map = {d: i for i, d in enumerate(digits)}
    K = len(digits)

    train_ds = Wrapped(train, label_map)
    test_ds  = Wrapped(test,  label_map)

    num_workers = int(getattr(cfg, "num_workers", 0))
    pin_memory  = bool(getattr(cfg, "pin_memory", False))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader, K

# -------------------------
# Loader factory
# -------------------------

def make_loaders(cfg: CFG):
    # ---- generate ----
    if cfg.dataset == "two_moons":
        X, y = two_moons(cfg.n_samples, cfg.noise, cfg.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # infer dims/classes
        cfg.in_dim = int(X.shape[1])
        cfg.num_classes = int(np.unique(y).size)

        # optional standardize
        if cfg.standardize:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mu) / sd

        # train/test split
        n = X.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(cfg.random_state)
        rng.shuffle(idx)

        n_test = int(round(cfg.test_ratio * n))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        return train_loader, test_loader

    elif cfg.dataset == "mnist":
        train_loader, test_loader, K = make_mnist_subset(cfg)

        # set dims/classes for downstream code
        # if your model expects flattened, keep in_dim=784; if it expects image, you can still store 784 here.
        cfg.in_dim = int(getattr(cfg, "mnist_in_dim", 28 * 28))
        cfg.num_classes = int(K)

        return train_loader, test_loader

    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")