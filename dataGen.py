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


# -------------------------
# Basic datasets
# -------------------------

def two_moons(n_samples=800, noise=0.12, random_state=42):
    return make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )


def circles(n_samples=800, noise=0.06, factor=0.5, random_state=42):
    return make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )


def blobs(n_samples=800, centers=3, cluster_std=1.2, random_state=42):
    return make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def classification_2d(
    n_samples=800,
    class_sep=1.0,
    flip_y=0.03,
    random_state=42,
):
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


def gaussian_quantiles(
    n_samples=800,
    n_classes=3,
    random_state=42,
):
    return make_gaussian_quantiles(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_classes,
        random_state=random_state,
    )


def hastie_10_2(n_samples=800, random_state=42, use_first2=True):
    X, y = make_hastie_10_2(
        n_samples=n_samples,
        random_state=random_state,
    )

    # map {-1,1} -> {0,1}
    y = (y > 0).astype(np.int64)

    if use_first2:
        X = X[:, :2]

    return X, y


# -------------------------
# Manifold datasets
# -------------------------

def swiss_roll(n_samples=800, noise=0.25, random_state=42, pca2=True):
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    if pca2:
        X = PCA(n_components=2, random_state=random_state).fit_transform(X)

    return X, t


def s_curve(n_samples=800, noise=0.25, random_state=42, pca2=True):
    X, t = make_s_curve(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    if pca2:
        X = PCA(n_components=2, random_state=random_state).fit_transform(X)

    return X, t


def make_loaders(cfg: CFG):
    # ---- generate ----
    if cfg.dataset == "two_moons":
        X, y = two_moons(cfg.n_samples, cfg.noise, cfg.random_state)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

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