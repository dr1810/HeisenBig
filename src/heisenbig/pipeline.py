"""Thin end-to-end pipeline: stage 1 (ETL) → 2 (grid + PCA) → 3 (MC UQ).

    python -m heisenbig.pipeline            # all ENDF files  → data/processed/
    python -m heisenbig.pipeline --quick    # 4 nuclides, small N → data/processed/quick/
    python -m heisenbig.pipeline --hdfs     # Parquet datasets on HDFS, small artifacts local

Outputs (under --out):
    xs_long/               Parquet, long schema, partitioned by nuclide   (stage 1)
    etl_errors.json        files that failed to parse
    xs_wide/               Parquet, one row per (nuclide, MT), xs array    (stage 2)
    xs_wide_grid.json      shared energy grid definition
    models/pca.joblib      fitted PCA (all components up to k_max)
    pca_errors.json        reconstruction error vs k, chosen k
    uq/samples/            Parquet MC samples, partitioned by method        (stage 3)
    uq_summary.json        nominal k_inf, MC mean/std, timings
    run_summary.json       per-stage stats and wall-clock seconds
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from heisenbig import compress, etl, grid, uq
from heisenbig.spark import get_spark

log = logging.getLogger("heisenbig.pipeline")

QUICK_NUCLIDES = ["U235", "U238", "H1", "O16", "Pu239", "Fe56", "Zr90", "B10"]


def run(raw_dir: str, out: str, nuclides=None, n_grid: int = 2000, n_samples: int = 2000,
        k: int | None = None, hdfs: bool = False, hdfs_root: str = "/heisenbig",
        uq_cfg: uq.UQConfig | None = None) -> dict:
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
    summary["stages"]["2_grid_compress"] = s2 | {
        "chosen_k": k_chosen, "matrix_shape": list(X.shape),
        "seconds": round(time.perf_counter() - t, 1)}

    # Stage 3 -------------------------------------------------------------
    t = time.perf_counter()
    cfg = uq_cfg or uq.UQConfig(n_samples=n_samples)
    s3 = uq.run_uq(spark, meta, X, g.energies, model, raw_dir, uri["uq/samples"],
                   out_dir / "uq_summary.json", cfg)
    summary["stages"]["3_uq"] = {
        "k_inf_original": s3["k_inf_nominal_original"],
        "k_inf_reconstructed": s3["k_inf_nominal_reconstructed"],
        "mc": s3["mc"], "seconds": round(time.perf_counter() - t, 1)}

    summary["total_seconds"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


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
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    nuclides = a.nuclides or (QUICK_NUCLIDES if a.quick else None)
    out = a.out or ("data/processed/quick" if a.quick else "data/processed")
    n_grid = a.n_grid or (500 if a.quick else 2000)
    n_samples = a.n_samples or (400 if a.quick else 2000)
    summary = run(a.raw_dir, out, nuclides, n_grid, n_samples, a.k, a.hdfs)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
