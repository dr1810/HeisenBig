"""Stage 2b — compress resampled curves into a low-dimensional latent space.

Features are ``log10(xs + EPS)`` (cross sections span ~10 decades). PCA is the
baseline compressor; anything implementing :class:`Compressor` (e.g. a PyTorch
autoencoder later) can replace it without touching the UQ stage, which only calls
``encode`` / ``decode`` and works in *linear* cross-section space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

EPS = 1e-10
K_CANDIDATES = (2, 4, 8, 16, 32, 64)


def to_features(xs: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(xs, 0.0, None) + EPS)


def from_features(f: np.ndarray) -> np.ndarray:
    return np.clip(10.0 ** f - EPS, 0.0, None)


class Compressor(Protocol):
    k: int

    def fit(self, xs: np.ndarray) -> "Compressor": ...
    def encode(self, xs: np.ndarray) -> np.ndarray: ...   # linear xs [n, N] -> z [n, k]
    def decode(self, z: np.ndarray) -> np.ndarray: ...    # z [n, k] -> linear xs [n, N]
    def save(self, path: str | Path) -> None: ...


class PCACompressor:
    """PCA in log10 feature space. Fitted once with ``k_max`` components;
    ``with_k`` returns a view truncated to the first k (PCA bases are nested)."""

    def __init__(self, k: int = 16, k_max: int | None = None, random_state: int = 0):
        self.k = k
        self.k_max = max(k, k_max or k)
        self.random_state = random_state
        self.pca = None

    def fit(self, xs: np.ndarray) -> "PCACompressor":
        from sklearn.decomposition import PCA

        n_comp = min(self.k_max, *xs.shape)
        self.pca = PCA(n_components=n_comp, random_state=self.random_state).fit(to_features(xs))
        self.k = min(self.k, n_comp)
        return self

    def with_k(self, k: int) -> "PCACompressor":
        c = PCACompressor(k=min(k, self.pca.n_components_), k_max=self.k_max, random_state=self.random_state)
        c.pca = self.pca
        return c

    @property
    def components(self) -> np.ndarray:   # [k, N], orthonormal rows
        return self.pca.components_[: self.k]

    @property
    def mean(self) -> np.ndarray:         # [N]
        return self.pca.mean_

    def encode(self, xs: np.ndarray) -> np.ndarray:
        return (to_features(xs) - self.mean) @ self.components.T

    def decode(self, z: np.ndarray) -> np.ndarray:
        return from_features(z @ self.components + self.mean)

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"k": self.k, "k_max": self.k_max, "pca": self.pca}, path)

    @classmethod
    def load(cls, path: str | Path) -> "PCACompressor":
        import joblib

        d = joblib.load(path)
        c = cls(k=d["k"], k_max=d["k_max"])
        c.pca = d["pca"]
        return c


def relative_errors(xs: np.ndarray, xs_rec: np.ndarray) -> np.ndarray:
    """Per-row relative L2 error in linear xs space: ||rec - xs|| / ||xs||."""
    num = np.linalg.norm(xs_rec - xs, axis=1)
    den = np.linalg.norm(xs, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def error_vs_k(model: PCACompressor, xs: np.ndarray, ks=K_CANDIDATES) -> list[dict]:
    rows = []
    for k in ks:
        if k > model.pca.n_components_:
            continue
        m = model.with_k(k)
        err = relative_errors(xs, m.decode(m.encode(xs)))
        rows.append({
            "k": k, "mean_rel_err": float(err.mean()), "median_rel_err": float(np.median(err)),
            "p90_rel_err": float(np.quantile(err, 0.9)), "max_rel_err": float(err.max()),
            "explained_var": float(m.pca.explained_variance_ratio_[:k].sum()),
        })
    return rows


def choose_k(rows: list[dict], target_median: float = 0.01) -> int:
    """Smallest k whose *median* per-curve relative error ≤ target, else the largest k."""
    for r in rows:
        if r["median_rel_err"] <= target_median:
            return r["k"]
    return rows[-1]["k"]
