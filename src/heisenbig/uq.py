"""Stage 3 — Monte-Carlo uncertainty quantification of an infinite-medium k_inf.

Figure of merit (one-group, homogeneous mixture):

    k_inf = Σ_i N_i <ν σ_f>_i / Σ_i N_i <σ_f + σ_c>_i,   <g> = ∫ g φ dE / ∫ φ dE

with a fixed analytic reference flux φ(E) on the shared grid. Σ_a = capture
(MT102) + fission (MT18); scattering does not enter k_inf.

Two sampling schemes, both distributed with Spark (one task per batch of samples,
seeded per batch → reproducible regardless of partitioning):

* ``latent_<name>`` — for each compressor (PCA, optionally the autoencoder),
  perturb the latent vectors of the reactor nuclides' MT18/MT102 rows, decode,
  evaluate k_inf (centred on that compressor's reconstructed curves).
* ``full``   — perturb log10(xs) directly on all N grid points with correlated
  Gaussian noise (squared-exponential kernel in log10 E), evaluate k_inf
  (centred on the original curves).

All noise levels are calibrated to the same RMS relative xs perturbation
(latent: empirically, since latent units are compressor-specific).

CAVEAT: the thin pipeline uses ENDF MF=3 data, which holds only a background in
the resolved resonance region, so absolute k_inf values are not physical yet.
"""
from __future__ import annotations

import json
import logging
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from heisenbig.compress import from_features, to_features
from heisenbig.grid import resample

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Material: homogenised LWR-like pin cell, atom densities in atoms/(barn·cm).
# Assumptions (cold, clad ignored):
#   UO2 fuel, 10.4 g/cm3, M≈270 g/mol → N_U = 10.4·0.6022/270 = 2.320e-2
#   4.0 % enrichment (atom fraction ≈ weight fraction to 1 %, fine here)
#   water 1.0 g/cm3 → N_H2O = 0.6022/18.015 = 3.343e-2  (H = 2×, O = 1×)
#   volume fractions: fuel 0.35, water 0.65 (moderator/fuel ≈ 1.9, typical PWR)
# ---------------------------------------------------------------------------
_F_FUEL, _F_MOD, _ENR, _N_U, _N_H2O = 0.35, 0.65, 0.04, 2.320e-2, 3.343e-2
DEFAULT_DENSITIES = {
    "U235": _F_FUEL * _ENR * _N_U,                         # 3.25e-4
    "U238": _F_FUEL * (1 - _ENR) * _N_U,                   # 7.80e-3
    "H1": _F_MOD * 2 * _N_H2O,                             # 4.35e-2
    "O16": _F_FUEL * 2 * _N_U + _F_MOD * _N_H2O,           # 3.80e-2
}
# Fallback ν̄ if MF=1/MT=452 cannot be read (thermal / fast-fission typical values).
NU_FALLBACK = {"U235": 2.43, "U238": 2.8}
FOM_MTS = (18, 102)


# ---------------------------------------------------------------------------
# Reference flux φ(E) per unit energy (illustrative LWR-like shape, NOT a
# transport solution):
#   thermal   : Maxwellian  M(E) = E/kT² · exp(-E/kT),  kT = 0.0253 eV (293.6 K)
#   slowing   : 1/E with a Westcott-style joining factor 1/(1+(5kT/E)^7) at the
#               thermal end and exp(-E/1 MeV) roll-off at the fast end
#   fission   : Watt spectrum χ(E) ∝ exp(-E/a)·sinh(√(bE)), a=0.988 MeV, b=2.249 /MeV
#               (U-235 thermal fission, ENDF/MCNP standard parameters)
# Each component is normalised on the grid, then mixed by flux fraction.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FluxSpec:
    kT: float = 0.0253
    frac_thermal: float = 0.25
    frac_slowing: float = 0.35
    frac_fission: float = 0.40
    watt_a: float = 0.988e6
    watt_b: float = 2.249e-6


def trapz_weights(e: np.ndarray) -> np.ndarray:
    """Trapezoid quadrature weights for ∫ f dE on a sorted grid."""
    w = np.zeros_like(e)
    d = np.diff(e)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def reference_flux(e: np.ndarray, spec: FluxSpec = FluxSpec()) -> np.ndarray:
    dE = trapz_weights(e)
    x = e / spec.kT
    thermal = x / spec.kT * np.exp(-np.minimum(x, 700))
    slowing = (1.0 / e) / (1.0 + (5.0 * spec.kT / e) ** 7) * np.exp(-e / 1e6)
    with np.errstate(over="ignore"):
        fission = np.exp(-e / spec.watt_a) * np.sinh(np.sqrt(spec.watt_b * e))
    parts = []
    for comp in (thermal, slowing, fission):
        parts.append(comp / (comp @ dE))
    phi = spec.frac_thermal * parts[0] + spec.frac_slowing * parts[1] + spec.frac_fission * parts[2]
    return phi


def collapse_weights(e: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """w such that <g> = g @ w  (flux-weighted one-group average)."""
    w = phi * trapz_weights(e)
    return w / w.sum()


def k_inf(sig_f: dict, sig_c: dict, nu: dict, densities: dict, w: np.ndarray) -> np.ndarray:
    """Infinite-medium multiplication factor.

    sig_f / sig_c: nuclide → xs array [..., N] (missing nuclide = 0). nu: nuclide →
    ν̄(E) array [N] or scalar. Leading dims (e.g. samples) broadcast through.
    """
    prod = 0.0
    absn = 0.0
    for nuc, n in densities.items():
        sf = sig_f.get(nuc)
        sc = sig_c.get(nuc)
        if sf is not None:
            prod = prod + n * ((np.asarray(nu.get(nuc, 0.0)) * sf) @ w)
            absn = absn + n * (sf @ w)
        if sc is not None:
            absn = absn + n * (sc @ w)
    return np.asarray(prod) / np.asarray(absn)


def read_nubar(raw_dir: str | Path, nuclide: str, energies: np.ndarray):
    """ν̄_total(E) from MF=1/MT=452 on the grid, or (constant, reason) fallback."""
    from heisenbig.etl import list_endf_files

    try:
        import endf

        (path,) = list_endf_files(raw_dir, [nuclide])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sec = endf.Material(path).section_data.get((1, 452))
        if sec is None or sec.get("LNU") != 2:
            raise ValueError(f"MF1/MT452 missing or not tabulated (LNU={sec and sec.get('LNU')})")
        return resample(sec["nu"], energies), "endf MF1/MT452"
    except Exception as exc:
        log.warning("ν̄ for %s: using constant %.2f (%s)", nuclide, NU_FALLBACK[nuclide], exc)
        return np.full_like(energies, NU_FALLBACK[nuclide]), f"constant fallback ({exc})"


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class UQConfig:
    n_samples: int = 2000
    batch_size: int = 250
    target_rel_pert: float = 0.0075      # RMS relative xs perturbation (0.75 %)
    corr_len_decades: float = 0.5        # full-space correlation length in log10(E)
    seed: int = 42
    densities: dict = field(default_factory=lambda: dict(DEFAULT_DENSITIES))
    flux: FluxSpec = field(default_factory=FluxSpec)


def _fom_from_rows(xs_rows: np.ndarray, ctx: dict) -> np.ndarray:
    """xs_rows [B, R, N] in ctx['rows'] order → k_inf [B]."""
    sig_f, sig_c = {}, {}
    for r, (nuc, mt) in enumerate(ctx["rows"]):
        (sig_f if mt == 18 else sig_c)[nuc] = xs_rows[:, r, :]
    return k_inf(sig_f, sig_c, ctx["nu"], ctx["densities"], ctx["w"])


def _rel_pert_rms(xs_pert: np.ndarray, xs_nom: np.ndarray) -> float:
    """RMS of xs_pert/xs_nom - 1 over points with a non-negligible nominal value
    (sub-threshold zeros decode to ~EPS, where a ratio is meaningless)."""
    mask = xs_nom > 1e-6
    rel = (xs_pert[mask] / xs_nom[mask]) - 1.0
    return float(np.sqrt(np.mean(rel ** 2)))


_METHOD_SEED = {"latent_pca": 0, "full": 1, "latent_ae": 2}


def run_batch(batch_id: int, ctx: dict, method: str) -> list[tuple]:
    """Worker-side: one batch of MC samples → [(method, sample_id, k_inf, rel_pert_rms)].

    ``latent_*`` methods perturb z0 + c · s ∘ ε (s = per-dimension spread of the
    training codes, c = calibrated scalar) and decode with ctx['model'] — any
    Compressor. ``full`` perturbs log10(xs) with correlated noise on the grid."""
    n0 = batch_id * ctx["batch_size"]
    b = min(ctx["batch_size"], ctx["n_samples"] - n0)
    code = _METHOD_SEED.get(method, sum(map(ord, method)))
    rng = np.random.default_rng([ctx["seed"], code, batch_id])
    R = len(ctx["rows"])
    if method.startswith("latent"):
        model = ctx["model"]
        if hasattr(model, "net"):          # torch model: one thread per Spark task, no oversubscription
            import torch
            torch.set_num_threads(1)
        z0 = ctx["z0"]                                       # [R, k]
        z = z0[None] + ctx["scale"] * ctx["code_std"] * rng.standard_normal((b, R, z0.shape[1]))
        xs = model.decode(z.reshape(b * R, -1)).reshape(b, R, -1)
        nom = ctx["xs_rec"]
    else:
        f0 = ctx["f0"]                                       # [R, N]
        eps = rng.standard_normal((b * R, f0.shape[1])) @ ctx["L"].T
        xs = from_features(f0[None] + ctx["sigma_f"] * eps.reshape(b, R, -1))
        nom = ctx["xs_orig"]
    k = _fom_from_rows(xs, ctx)
    rel = [_rel_pert_rms(xs[i], nom) for i in range(b)]
    return [(method, n0 + i, float(k[i]), rel[i]) for i in range(b)]


def corr_factor(log10_e: np.ndarray, corr_len: float) -> np.ndarray:
    """L with L Lᵀ = squared-exponential correlation matrix (via eigh, PSD-clipped)."""
    d = log10_e[:, None] - log10_e[None, :]
    C = np.exp(-0.5 * (d / corr_len) ** 2)
    lam, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(lam, 0.0, None))


def calibrate_latent_scale(model, z0: np.ndarray, code_std: np.ndarray, xs_rec: np.ndarray,
                           target: float, seed: int = 0, n_cal: int = 32, iters: int = 5):
    """Scalar c so that decoding z0 + c·s∘ε gives ~``target`` RMS relative xs
    perturbation (fixed-point iteration on a fixed calibration draw). Works for any
    compressor, whose latent units are otherwise arbitrary."""
    rng = np.random.default_rng([seed, 99])
    eps = rng.standard_normal((n_cal,) + z0.shape)
    c, measured = 0.01, 0.0
    for _ in range(iters):
        xs = model.decode((z0[None] + c * code_std * eps).reshape(-1, z0.shape[1])).reshape(n_cal, *xs_rec.shape)
        measured = float(np.mean([_rel_pert_rms(x, xs_rec) for x in xs]))
        if measured <= 0:
            c *= 10
            continue
        c *= target / measured
    return c, measured


def run_uq(spark, meta, X: np.ndarray, energies: np.ndarray, compressors, raw_dir: str | Path,
           out_uri: str, summary_path: str | Path, cfg: UQConfig = UQConfig(),
           code_std: dict | None = None) -> dict:
    """meta/X: wide table (from grid.load_wide).

    compressors: a fitted Compressor, or {name: Compressor} (e.g. {"pca": …, "ae": …}).
    code_std: {name: per-dimension std of training codes}; defaults to the std of
    the codes of all rows in X.
    """
    if not isinstance(compressors, dict):
        compressors = {"pca": compressors}
    rows = [(n, mt) for n in cfg.densities for mt in FOM_MTS
            if ((meta["nuclide"] == n) & (meta["mt"] == mt)).any()]
    missing = [n for n in cfg.densities if n not in set(meta["nuclide"])]
    if missing:
        raise ValueError(f"reactor nuclides missing from wide table: {missing}")
    idx = [int(np.flatnonzero((meta["nuclide"] == n) & (meta["mt"] == mt))[0]) for n, mt in rows]
    xs_orig = X[idx]                                       # [R, N]

    nu, nu_src = {}, {}
    for n in cfg.densities:
        if n in NU_FALLBACK:
            nu[n], nu_src[n] = read_nubar(raw_dir, n, energies)

    phi = reference_flux(energies, cfg.flux)
    w = collapse_weights(energies, phi)
    base = {
        "rows": rows, "nu": nu, "densities": cfg.densities, "w": w,
        "batch_size": cfg.batch_size, "n_samples": cfg.n_samples, "seed": cfg.seed,
        "xs_orig": xs_orig, "f0": to_features(xs_orig),
        # full space: d xs/xs ≈ ln10 · d log10 xs
        "sigma_f": cfg.target_rel_pert / math.log(10),
    }
    k_orig = float(_fom_from_rows(xs_orig[None], base)[0])
    log.info("nominal k_inf (original curves): %.5f", k_orig)

    method_ctx, k_rec, latent_info = {}, {}, {}
    for name, model in compressors.items():
        z0 = model.encode(xs_orig)
        xs_rec = model.decode(z0)
        s = (code_std or {}).get(name)
        if s is None:
            s = model.encode(X).std(axis=0)
        s = np.where(s > 0, s, 1.0)
        c, measured = calibrate_latent_scale(model, z0, s, xs_rec, cfg.target_rel_pert, cfg.seed)
        k_rec[name] = float(_fom_from_rows(xs_rec[None], base)[0])
        latent_info[name] = {"k": int(model.k), "scale": c, "calibrated_rel_pert": measured}
        method_ctx[f"latent_{name}"] = base | {"model": model, "z0": z0, "xs_rec": xs_rec,
                                               "code_std": s, "scale": c}
        log.info("nominal k_inf (%s, k=%d): %.5f | latent scale %.4g", name, model.k, k_rec[name], c)
    method_ctx["full"] = base  # L added inside the timed section below

    from pyspark.sql import types as T

    sc = spark.sparkContext
    n_batches = math.ceil(cfg.n_samples / cfg.batch_size)
    results, dfs = {}, []
    schema = T.StructType([
        T.StructField("method", T.StringType()), T.StructField("sample_id", T.IntegerType()),
        T.StructField("k_inf", T.DoubleType()), T.StructField("rel_pert_rms", T.DoubleType()),
    ])
    for method, mctx in method_ctx.items():
        t0 = time.perf_counter()
        if method == "full":   # building the N×N factor is part of full-space cost
            mctx = mctx | {"L": corr_factor(np.log10(energies), cfg.corr_len_decades)}
        bc = sc.broadcast(mctx)
        out = (sc.parallelize(range(n_batches), n_batches)
                 .flatMap(lambda b, m=method, bc=bc: run_batch(b, bc.value, m))
                 .collect())
        seconds = time.perf_counter() - t0
        bc.unpersist()
        ks = np.array([r[2] for r in out])
        results[method] = {
            "mean": float(ks.mean()), "std": float(ks.std(ddof=1)),
            "rel_pert_rms_mean": float(np.mean([r[3] for r in out])),
            "seconds": round(seconds, 2),
        }
        dfs.append(spark.createDataFrame(out, schema))
        log.info("%s MC: %s", method, results[method])

    union = dfs[0]
    for d in dfs[1:]:
        union = union.unionByName(d)
    union.write.mode("overwrite").partitionBy("method").parquet(out_uri)

    summary = {
        "k_inf_nominal_original": k_orig, "k_inf_nominal_reconstructed": k_rec,
        "latent": latent_info, "perturbed_rows": [f"{n}/MT{mt}" for n, mt in rows],
        "nu_source": nu_src, "sigma_log10_full": base["sigma_f"],
        "config": asdict(cfg) | {"flux": asdict(cfg.flux)}, "mc": results,
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=1))
    return summary
