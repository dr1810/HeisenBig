# HeisenBig

Higher-dimensionality uncertainty quantification on ENDF/B-VIII.1 nuclear cross-section data.
This is the final project for NYU CS-GY Big Data, Fall 2026. The pipeline is Spark ETL → Ray/PyTorch compression → Spark Monte-Carlo UQ.

Background, conventions and the `endf` API cheat sheet are in **[AGENTS.md](AGENTS.md)**.

## Quick start

Requirements: Docker Desktop (or Docker Engine + Compose v2). The setup works natively on arm64 and amd64.

```bash
# 1. Data (not in git)
bash scripts/download_endf.sh          # ENDF/B-VIII.1 neutron sublibrary: 558 files, ~1.3 GB unpacked
bash scripts/download_ace_sample.sh    # optional: 80198.710nc ACE sample (streams a 3.5 GB archive)

# 2. Environment
docker compose up -d --build           # first build ~10 min; image ~4 GB
open http://localhost:8888             # JupyterLab, no token

# 3. Verify
docker compose exec dev pytest -q      # endf / Spark / PyTorch / Ray smoke tests
docker compose exec -w /workspace/notebooks dev \
  jupyter nbconvert --to notebook --execute --inplace 00_endf_docs_smoke_test.ipynb

# Optional: single-node HDFS (namenode:9000, UI http://localhost:9870)
docker compose --profile hdfs up -d
```

In code, use `from heisenbig.spark import get_spark` and call `get_spark(hdfs=True)` for HDFS.

| Port | Service |
|---|---|
| 8888 | JupyterLab |
| 4040 | Spark UI (while a session is alive) |
| 8265 | Ray dashboard |
| 9870 | HDFS NameNode UI (`hdfs` profile) |

To add a Python dependency, pin it in `docker/requirements.txt`, then run `docker compose build`.

## Thin end-to-end pipeline

A minimal version of all four stages. It runs on the ENDF files from `scripts/download_endf.sh` and needs nothing else.

| Stage | Module | What it does |
|---|---|---|
| 1 ETL | `heisenbig.etl` | Spark parses each ENDF file in a worker and writes MF=3 σ(E) for MT 1/2/18/102 to long Parquet `xs_long/`, partitioned by nuclide |
| 2 Grid + compression | `heisenbig.grid`, `heisenbig.compress` | Resamples every curve onto a shared 2000-point log grid (1e-5 eV–20 MeV) with ENDF interpolation laws, writes `xs_wide/`, and fits PCA on log10(σ) |
| 3 UQ | `heisenbig.uq` | One-group k∞ of a homogenised 4 % UO₂ + water mix, with Monte Carlo in PCA latent space vs full grid space, distributed with Spark |
| 4 Analysis | `notebooks/01_thin_e2e.ipynb` | Plots the saved outputs: curves, PCA error vs k, and k∞ histograms |

```bash
bash scripts/run_thin_e2e.sh --quick   # 8 nuclides, N=500, 400 samples → data/processed/quick/  (~20 s)
bash scripts/run_thin_e2e.sh           # all 558 files, N=2000, 2000 samples → data/processed/  (~1.5 min)
bash scripts/run_thin_e2e.sh --help    # --nuclides, --n-grid, --n-samples, --k, --hdfs, --out

docker compose exec dev pytest -q tests/test_pipeline.py
docker compose exec -w /workspace/notebooks dev \
  jupyter nbconvert --to notebook --execute --inplace 01_thin_e2e.ipynb
```

The script runs inside the dev container; it calls `docker compose exec` for you when started on the host.
With `--hdfs`, start HDFS first and wait for it: `docker compose --profile hdfs up -d && docker compose exec namenode hdfs dfsadmin -safemode wait`.

**Limitations of this version**
- **Absolute k∞ is not physical yet.** ENDF MF=3 holds only a background in the resolved resonance region. U-235 fission and capture are exactly zero below 2.24 keV, so thermal fission is missing entirely. Dense pointwise data (POINT2025 or ACE) will replace MF=3 later.
- **ν̄** is read from ENDF MF=1/MT=452 for U-235 and U-238. The constants 2.43 and 2.8 are only used if that read fails; they are not used in the current runs.
- **The flux spectrum is an analytic shape**, not a transport solution. Details and the decisions log are in [AGENTS.md](AGENTS.md) §8.
