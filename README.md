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
