"""Stage 2b (option) — masked MLP autoencoder compressor.

Why not just a plain autoencoder on log10(xs)? The PCA baseline fails mostly on
exact zeros: below thresholds and across the empty resolved-resonance region,
log10(0 + EPS) = -10, so every curve has cliffs of ~10 decades at different
energies. This model separates *where* a cross section exists from *how large*
it is:

    input   [N log-values (standardised, 0 where xs == 0) | N presence mask]
    encoder 2N → 512 → 256 → k
    decoder k → 256 → 512 → 2N  = [N mask logits | N standardised log-values]
    loss    BCE(mask) + MSE(log-value, only where xs > 0)
    decode  xs = 10^value where logit > 0, else exactly 0

Implements the :class:`heisenbig.compress.Compressor` interface, so stage 3 can
use it interchangeably with PCA. Several latent sizes k are trained in parallel
as Ray tasks (one CPU slice per k).
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from heisenbig.compress import relative_errors, to_features

log = logging.getLogger(__name__)


@dataclass
class AEConfig:
    hidden: tuple = (512, 256)
    epochs: int = 150
    batch_size: int = 64
    lr: float = 2e-3            # OneCycle peak
    weight_decay: float = 1e-4
    eval_every: int = 10        # epochs between validation checks (best checkpoint kept)
    seed: int = 0


def _build(n: int, k: int, hidden):
    import torch.nn as nn

    h0, h1 = hidden
    enc = nn.Sequential(nn.Linear(2 * n, h0), nn.GELU(), nn.Linear(h0, h1), nn.GELU(), nn.Linear(h1, k))
    dec = nn.Sequential(nn.Linear(k, h1), nn.GELU(), nn.Linear(h1, h0), nn.GELU(), nn.Linear(h0, 2 * n))
    return nn.ModuleDict({"enc": enc, "dec": dec})


class AECompressor:
    """Masked autoencoder; ``encode``/``decode`` work on linear xs, like PCACompressor."""

    def __init__(self, n: int, k: int, cfg: AEConfig = AEConfig()):
        import torch

        self.n, self.k, self.cfg = n, k, cfg
        torch.manual_seed(cfg.seed)          # weight init must not depend on process RNG state
        self.net = _build(n, k, cfg.hidden)
        # feature normalisation, set by fit(): mean/std/min/max of log10(xs) where xs > 0
        self.mu = 0.0
        self.sd = 1.0
        self.fmin = -10.0
        self.fmax = 10.0

    # ----- feature plumbing ------------------------------------------------
    def _inputs(self, xs: np.ndarray):
        import torch

        mask = (xs > 0).astype(np.float32)
        val = np.where(mask > 0, (to_features(xs) - self.mu) / self.sd, 0.0).astype(np.float32)
        return torch.from_numpy(np.concatenate([val, mask], axis=1))

    def encode(self, xs: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            return self.net["enc"](self._inputs(np.atleast_2d(xs))).numpy().astype(float)

    def decode(self, z: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net["dec"](torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)).numpy()
        logit, val = out[:, : self.n], out[:, self.n:]
        f = np.clip(val * self.sd + self.mu, self.fmin, self.fmax)   # never extrapolate beyond training range
        return np.where(logit > 0, 10.0 ** f, 0.0)

    # ----- training --------------------------------------------------------
    def fit(self, xs_train: np.ndarray, xs_val: np.ndarray | None = None, threads: int | None = None) -> dict:
        import torch
        import torch.nn.functional as F

        if threads:
            torch.set_num_threads(threads)
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        f, on = to_features(xs_train), xs_train > 0
        self.mu, self.sd = float(f[on].mean()), float(f[on].std() or 1.0)
        self.fmin, self.fmax = float(f[on].min()), float(f[on].max())

        x_tr = self._inputs(xs_train)
        n_tr, n = len(xs_train), self.n
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr / 10, weight_decay=self.cfg.weight_decay)
        steps_per_epoch = int(np.ceil(n_tr / self.cfg.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.cfg.lr, total_steps=self.cfg.epochs * steps_per_epoch)

        best = (np.inf, None, -1)
        history = []
        t0 = time.perf_counter()
        for epoch in range(1, self.cfg.epochs + 1):
            self.net.train()
            for b in torch.from_numpy(rng.permutation(n_tr)).split(self.cfg.batch_size):
                x = x_tr[b]
                mask = x[:, n:]
                out = self.net["dec"](self.net["enc"](x))
                loss = (F.binary_cross_entropy_with_logits(out[:, :n], mask)
                        + ((out[:, n:] - x[:, :n]) ** 2 * mask).sum() / mask.sum().clamp(min=1.0))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs:
                ref = xs_val if xs_val is not None and len(xs_val) else xs_train
                err = relative_errors(ref, self.decode(self.encode(ref)))
                score = float(np.median(err))
                history.append({"epoch": epoch, "loss": float(loss), "val_median_rel_err": score})
                if score < best[0]:
                    best = (score, {k: v.clone() for k, v in self.net.state_dict().items()}, epoch)
        self.net.load_state_dict(best[1])
        return {"k": self.k, "best_epoch": best[2], "seconds": round(time.perf_counter() - t0, 1),
                "history": history}

    # ----- persistence -----------------------------------------------------
    def _state(self) -> dict:
        return {"n": self.n, "k": self.k, "cfg": asdict(self.cfg), "state_dict": self.net.state_dict(),
                "norm": (self.mu, self.sd, self.fmin, self.fmax)}

    @classmethod
    def _from_state(cls, d: dict) -> "AECompressor":
        cfg = AEConfig(**{**d["cfg"], "hidden": tuple(d["cfg"]["hidden"])})
        m = cls(d["n"], d["k"], cfg)
        m.net.load_state_dict(d["state_dict"])
        m.mu, m.sd, m.fmin, m.fmax = d["norm"]
        return m

    def save(self, path: str | Path) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state(), path)

    @classmethod
    def load(cls, path: str | Path) -> "AECompressor":
        import torch

        return cls._from_state(torch.load(path, weights_only=False))

    # Spark/Ray pickling: serialise via torch so the object travels as plain bytes.
    def __getstate__(self):
        import torch

        buf = io.BytesIO()
        torch.save(self._state(), buf)
        return {"blob": buf.getvalue()}

    def __setstate__(self, s):
        import torch

        other = self._from_state(torch.load(io.BytesIO(s["blob"]), weights_only=False))
        self.__dict__.update(other.__dict__)


def _train_one(xs_train, xs_val, k, cfg, threads):
    m = AECompressor(xs_train.shape[1], k, cfg)
    info = m.fit(xs_train, xs_val, threads=threads)
    return m, info


def _ray_sweep(xs_train, xs_val, ks, cfg, num_cpus, threads) -> dict:
    import ray

    # small object store: fits in the container's 2 GB /dev/shm
    ray.init(num_cpus=num_cpus, include_dashboard=False, log_to_driver=False,
             object_store_memory=512 * 1024 * 1024)
    try:
        remote = ray.remote(num_cpus=threads)(_train_one)
        tr_ref, va_ref = ray.put(xs_train), ray.put(xs_val)
        results = ray.get([remote.remote(tr_ref, va_ref, k, cfg, threads) for k in ks])
    finally:
        ray.shutdown()
    return dict(zip(ks, results))


def train_sweep(xs_train: np.ndarray, xs_val: np.ndarray, ks, cfg: AEConfig = AEConfig(),
                use_ray: bool = True, num_cpus: int | None = None) -> dict:
    """Train one AE per k (in parallel with Ray). Returns {k: (AECompressor, info)}.

    Ray runs in a separate *spawned* process: starting Ray inside the Spark driver
    process breaks its Py4J gateway connections (observed: ConnectionResetError on
    the next Spark call, even without ray.shutdown())."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    ks = list(ks)
    num_cpus = num_cpus or os.cpu_count() or 1
    threads = max(1, num_cpus // len(ks))
    if not use_ray or len(ks) == 1:
        return {k: _train_one(xs_train, xs_val, k, cfg, threads=num_cpus) for k in ks}
    log.info("AE sweep: k=%s on Ray (child process), %d threads per trial", ks, threads)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_ray_sweep, xs_train, xs_val, ks, cfg, num_cpus, threads).result()
