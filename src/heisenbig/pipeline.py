"""Thin end-to-end pipeline: stage 1 (ETL) → 2 (grid + PCA) → 3 (MC UQ).

    python -m heisenbig.pipeline            # all ENDF files  → data/processed/
    python -m heisenbig.pipeline --quick    # 4 nuclides, small N → data/processed/quick/
    python -m heisenbig.pipeline --hdfs     # Parquet datasets on HDFS, small artifacts local
    python -m heisenbig.pipeline --compressor ae   # + masked autoencoder (Ray sweep over k)

Outputs (under --out):
    xs_long/               Parquet, long schema, partitioned by nuclide   (stage 1)
    etl_errors.json        files that failed to parse
    xs_wide/               Parquet, one row per (nuclide, MT), xs array    (stage 2)
    xs_wide_grid.json      shared energy grid definition
    models/pca.joblib      fitted PCA (all components up to k_max)
    pca_errors.json        reconstruction error vs k (in-sample, all rows), chosen k
    compressor_comparison.json   (--compressor ae) PCA vs AE on a shared held-out split
    models/ae.pt           (--compressor ae) autoencoder at the chosen k
    uq/samples/            Parquet MC samples, partitioned by method        (stage 3)
    uq_summary.json        nominal k_inf, MC mean/std, timings
    run_summary.json       per-stage stats and wall-clock seconds
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from heisenbig import autoencoder, compress, etl, grid, uq
from heisenbig.spark import get_spark

log = logging.getLogger("heisenbig.pipeline")

QUICK_NUCLIDES = ["U235", "U238", "H1", "O16", "Pu239", "Fe56", "Zr90", "B10"]


def run(raw_dir: str, out: str, nuclides=None, n_grid: int = 2000, n_samples: int = 2000,
        k: int | None = None, hdfs: bool = False, hdfs_root: str = "/heisenbig",
        uq_cfg: uq.UQConfig | None = None, compressor: str = "pca",
        ae_cfg: autoencoder.AEConfig | None = None, ae_ks=(8, 16, 32, 64)) -> dict:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet datasets go to HDFS with --hdfs; JSON/model artifacts always stay local.
    base = f"{hdfs_root}/{out_dir.name}" if hdfs else str(out_dir.resolve())
    uri = {name: f"{base}/{name}" for name in ("xs_long", "xs_wide", "uq/samples")}

    spark = get_spark("heisenbig-thin-e2e", hdfs=hdfs)
    spark.sparkContext.setLogLevel("ERROR")
    summary: dict = {"out": str(out_dir), "hdfs": hdfs, "uris": uri, "stages": {}}
    t_all = time.perf_counter()

    # Stage 1 -------------------------------------------------------------
    t = time.perf_counter()
    s1 = etl.run_etl(spark, raw_dir, uri["xs_long"], nuclides, error_log=out_dir / "etl_errors.json")
    summary["stages"]["1_etl"] = s1 | {"seconds": round(time.perf_counter() - t, 1)}

    # Stage 2 -------------------------------------------------------------
    t = time.perf_counter()
    g = grid.Grid(n=n_grid)
    s2 = grid.resample_long(spark, uri["xs_long"], uri["xs_wide"], g, out_dir / "xs_wide_grid.json")
    meta, X = grid.load_wide(spark, uri["xs_wide"])
    model = compress.PCACompressor(k=max(compress.K_CANDIDATES), k_max=max(compress.K_CANDIDATES)).fit(X)
    errs = compress.error_vs_k(model, X)
    k_chosen = k or compress.choose_k(errs)
    model = model.with_k(k_chosen)
    model.save(out_dir / "models" / "pca.joblib")
    (out_dir / "pca_errors.json").write_text(json.dumps({"chosen_k": k_chosen, "rows": errs}, indent=1))
    compressors = {"pca": model}
    s2_extra: dict = {}
    if compressor != "ae":   # don't leave a previous AE run's artifacts next to these outputs
        for stale in (out_dir / "compressor_comparison.json", out_dir / "models" / "ae.pt"):
            stale.unlink(missing_ok=True)
    else:
        compressors["ae"], s2_extra = _fit_autoencoder(meta, X, out_dir, ae_cfg or autoencoder.AEConfig(),
                                                       ae_ks, k, (uq_cfg or uq.UQConfig()).densities)
    summary["stages"]["2_grid_compress"] = s2 | {
        "chosen_k": k_chosen, "matrix_shape": list(X.shape), **s2_extra,
        "seconds": round(time.perf_counter() - t, 1)}

    # Stage 3 -------------------------------------------------------------
    t = time.perf_counter()
    cfg = uq_cfg or uq.UQConfig(n_samples=n_samples)
    s3 = uq.run_uq(spark, meta, X, g.energies, compressors, raw_dir, uri["uq/samples"],
                   out_dir / "uq_summary.json", cfg)
    summary["stages"]["3_uq"] = {
        "k_inf_original": s3["k_inf_nominal_original"],
        "k_inf_reconstructed": s3["k_inf_nominal_reconstructed"],
        "mc": s3["mc"], "seconds": round(time.perf_counter() - t, 1)}

    summary["total_seconds"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _fit_autoencoder(meta, X, out_dir: Path, cfg, ks, k_forced, reactor_nuclides):
    """Shared held-out split → PCA and AE at each k → comparison table → AE at chosen k."""
    tr, va = compress.train_val_split(meta["nuclide"].to_numpy(), force_train=list(reactor_nuclides))
    in_sample = len(va) == 0
    if in_sample:   # e.g. --nuclides with only the reactor nuclides: nothing left to hold out
        log.warning("no held-out curves (all rows are forced into training); comparing in-sample")
        va = tr
    n_max = min(len(tr), X.shape[1])
    ks = [kk for kk in ks if kk <= n_max] or [min(ks[0], n_max)]
    rows = []
    for kk in ks:   # PCA refit on the training split only, so the comparison is fair
        p = compress.PCACompressor(k=kk, k_max=kk).fit(X[tr])
        rows.append(compress.heldout_row("pca", kk, p, X[tr], X[va]))
    t = time.perf_counter()
    trained = autoencoder.train_sweep(X[tr], X[va], ks, cfg)
    sweep_s = round(time.perf_counter() - t, 1)
    ae_rows = []
    for kk, (m, info) in trained.items():
        ae_rows.append(compress.heldout_row("ae", kk, m, X[tr], X[va], best_epoch=info["best_epoch"],
                                            train_seconds=info["seconds"]))
    k_ae = k_forced if k_forced in trained else compress.choose_k_tolerance(ae_rows, "val_median_rel_err")
    ae = trained[k_ae][0]
    ae.save(out_dir / "models" / "ae.pt")
    (out_dir / "compressor_comparison.json").write_text(json.dumps({
        "split": {"train": len(tr), "val": 0 if in_sample else len(va), "in_sample": in_sample,
                  "forced_train": list(reactor_nuclides)},
        "ae_config": dataclasses.asdict(cfg), "ae_chosen_k": k_ae, "ae_sweep_seconds": sweep_s,
        "rows": rows + ae_rows}, indent=1))
    log.info("AE sweep done in %.0f s, chosen k=%d", sweep_s, k_ae)
    return ae, {"ae_chosen_k": k_ae, "ae_sweep_seconds": sweep_s}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw/endf-b8.1/neutrons")
    ap.add_argument("--out", default=None, help="default: data/processed (quick: data/processed/quick)")
    ap.add_argument("--quick", action="store_true",
                    help=f"only {QUICK_NUCLIDES}, N_grid=500, 400 samples")
    ap.add_argument("--nuclides", nargs="*", help="limit stage 1 to these nuclides")
    ap.add_argument("--n-grid", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="force latent dimension (default: auto)")
    ap.add_argument("--hdfs", action="store_true", help="write Parquet datasets to HDFS")
    ap.add_argument("--compressor", choices=["pca", "ae"], default="pca",
                    help="ae: also train the masked autoencoder and compare it with PCA")
    ap.add_argument("--ae-epochs", type=int, default=None, help="default 150 (quick: 40)")
    ap.add_argument("--ae-ks", type=int, nargs="*", default=None, help="default 8 16 32 64 (quick: 4 8)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    nuclides = a.nuclides or (QUICK_NUCLIDES if a.quick else None)
    out = a.out or ("data/processed/quick" if a.quick else "data/processed")
    n_grid = a.n_grid or (500 if a.quick else 2000)
    n_samples = a.n_samples or (400 if a.quick else 2000)
    ae_cfg = autoencoder.AEConfig(epochs=a.ae_epochs or (40 if a.quick else 150))
    ae_ks = a.ae_ks or ([4, 8] if a.quick else [8, 16, 32, 64])
    summary = run(a.raw_dir, out, nuclides, n_grid, n_samples, a.k, a.hdfs,
                  compressor=a.compressor, ae_cfg=ae_cfg, ae_ks=ae_ks)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
