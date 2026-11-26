# AGENTS.md — HeisenBig project context

Shared context for AI coding agents (Claude Code, Codex, Cursor, …) and new teammates.
Read this before writing code. Keep it current when decisions change.

---

## 1. Project in one paragraph

**Higher-Dimensionality Uncertainty Quantification in Nuclear Reactor Physics** — NYU CS-GY Big Data,
Fall 2026 final project. Team: Dharun Ramesh, Seho Kim.

Evaluated nuclear cross sections (ENDF/B-VIII.1) are tabulated on very fine energy grids; treating
every energy point as an independent uncertain input gives UQ problems with up to millions of
dimensions. We (1) ETL the cross-section data with Spark into Parquet on HDFS, (2) learn a compact
non-linear latent representation with distributed PyTorch / Ray, (3) run Monte-Carlo UQ in that
latent space with Spark, and (4) analyse how perturbations (e.g. "+0.5 % at energy E") propagate to
safety-relevant quantities.

```
[1] ENDF → Spark ETL → Parquet/HDFS  →  [2] Deep compression (Ray / DDP PyTorch)
      →  [3] Latent-space Monte-Carlo UQ (Spark)  →  [4] Final analysis
```

---

## 2. Repository layout

```
AGENTS.md              ← this file (CLAUDE.md just points here)
docker-compose.yml     ← shared environment (dev container + optional HDFS)
docker/                ← Dockerfile, pinned requirements, HDFS image/config
scripts/               ← data download scripts; run_thin_e2e.sh (pipeline entry point)
notebooks/             ← 00_endf_docs_smoke_test (env check), 01_thin_e2e (stage 4 analysis)
src/heisenbig/         ← importable project code (PYTHONPATH=/workspace/src in container)
  spark.py             ← get_spark(hdfs=...) SparkSession factory
  etl.py               ← stage 1: ENDF files → long Parquet (Spark, one task per file)
  grid.py              ← stage 2a: shared log grid + resampling via Tabulated1D (applyInPandas)
  compress.py          ← stage 2b: Compressor interface (encode/decode) + PCA baseline, split helpers
  autoencoder.py       ← stage 2b option: masked MLP autoencoder + Ray sweep over k (--compressor ae)
  uq.py                ← stage 3: k_inf figure of merit, reference flux, Spark MC (latent vs full)
  pipeline.py          ← `python -m heisenbig.pipeline [--quick] [--hdfs]` runs stages 1→3
tests/                 ← pytest; test_environment.py (env), test_pipeline.py (pipeline)
docs/                  ← longer background notes
data/                  ← git-ignored. raw/, processed/, downloads/
```

Data layout (after running the scripts):

| Path | Content |
|---|---|
| `data/raw/endf-b8.1/neutrons/n-ZZZ_Sy_AAA.endf` | 558 ENDF-6 neutron sublibrary files (~1.3 GB) |
| `data/raw/ace/80198.710nc` | One sample ACE table (Hg-198, ENDF/B-VII.1, 293.6 K) |
| `data/processed/` | Full pipeline outputs (see `pipeline.py` docstring for the file list) |
| `data/processed/quick/` | `--quick` run outputs (kept separate so they never overwrite the full run) |

---

## 3. Environment (everyone uses the same container)

```bash
bash scripts/download_endf.sh          # ENDF/B-VIII.1 neutrons (~344 MB download)
bash scripts/download_ace_sample.sh    # optional: one ACE table for the ACE examples
docker compose up -d --build           # JupyterLab → http://localhost:8888 (no token)
docker compose exec dev bash           # shell inside the container
docker compose --profile hdfs up -d    # + HDFS (namenode:9000, UI http://localhost:9870)
docker compose exec dev pytest -q      # environment smoke tests
```

Stack inside `heisenbig/dev` (pinned in `docker/requirements.txt`):
Python 3.12 · OpenJDK 17 · PySpark 4.3 (local mode) · PyTorch 2.15 CPU · Ray 2.60 · endf 0.1.12 ·
numpy/scipy/pandas/pyarrow/matplotlib/seaborn · scikit-learn (PCA/SVD baselines) · tqdm · JupyterLab. Repo is bind-mounted at `/workspace`.

Rules for agents:
- **Run code inside the container** (`docker compose exec dev …`), not on the host — host Python
  versions differ.
- Add a dependency by pinning it in `docker/requirements.txt` and rebuilding; never `pip install`
  ad hoc in notebooks.
- Never commit anything under `data/`. Use the download scripts.
- Spark local mode is the default; for HDFS use `src/heisenbig/spark.py::get_spark(hdfs=True)` which
  sets `dfs.client.use.datanode.hostname=true` (required across containers).
- The Docker Desktop VM has ~12 GB RAM on the reference machine — don't load all 558 materials into
  one driver process; parallelise per file with Spark.

---

## 4. Physics background (just enough to not make mistakes)

**Microscopic cross section σ(E)** — effective area (unit **barn**, 1 b = 1e-24 cm²) describing the
probability that a neutron of incident energy **E** undergoes a given reaction with a nucleus.
Depends on target nuclide, reaction channel, energy, and temperature (Doppler broadening).

**Energy range**: ~1e-5 eV (thermal tail) → 20 MeV (up to 150–200 MeV for some evaluations).
Always plot on log-log axes. All ENDF energies are in **eV**.

**Energy regions**
1. *Thermal* (< ~1 eV): smooth, often ∝ 1/v for capture.
2. *Resolved resonance region (RRR)*: sharp, narrow peaks (resonances) where E matches a compound
   nucleus level. U-235 RRR extends to a few keV; U-238 to ~20 keV. Peaks can span 3–4 orders of
   magnitude → this is where "millions of points" come from after reconstruction.
3. *Unresolved resonance region (URR)*: resonances overlap/unmeasurable; given as averaged
   parameters + probability tables.
4. *Fast / continuum*: smooth, model-based.

**Key point for this project**: in an ENDF-6 file, resonances are stored as **parameters (MF=2)**,
not as points. MF=3 only contains a smooth *background* in the RRR. A dense pointwise σ(E) requires
resonance reconstruction (+ Doppler broadening), as done by NJOY (RECONR/BROADR) — the `endf`
package does **not** do this. Pointwise sources:
- NNDC **ENDF/B-VIII.1 POINT2025** library (pre-reconstructed, several temperatures) —
  https://www.nndc.bnl.gov/endf-library/B-VIII.1/POINT2025 — likely the best dense source for us.
- **ACE** files (LANL Lib81 for VIII.1, ~9.4 GB; ENDF71x for VII.1) — already reconstructed and
  broadened; readable with `endf.ace`.
- OpenMC's `openmc.data` can also reconstruct RRR from MF=2 (not installed yet).

**Uncertainty/covariance data**: ENDF provides covariance files — MF=33 (cross-section covariances,
on coarse energy groups), MF=32 (resonance parameter covariances), MF=31 (ν̄), MF=34/35
(angular/energy distributions). These are the physically motivated priors for UQ sampling.
Verified with endf 0.1.12 on VIII.1 U-235: **MF=33 is parsed** into `section_data[33, MT]`
(87 sections, keys `ZA, AWR, MTL, NL, subsections`); **MF=31 and MF=35 are NOT parsed** (UserWarning
"ignored") — only `section_text` is available for those, so parse them ourselves if needed.

Verified scale gap: U-235 (n,γ) in ENDF MF=3 has only **111 points**; the reconstructed ACE Hg-198
(n,γ) has **3,180** points on its grid. Use a pointwise source (below) for the dense data.

**Quantities of interest for UQ** (stage 4): k_eff (criticality/reactivity margin), reaction rates,
power peaking. A cheap proxy without a transport code: one-group / few-group collapsed cross
sections with a fixed flux spectrum (e.g. thermal Maxwellian + 1/E + fission spectrum), and
infinite-medium k∞ = ν Σ_f / Σ_a.

---

## 5. ENDF-6 format essentials

Fixed-width 80-column text. Each line: 6 × 11-char fields, then MAT (4), MF (2), MT (3), line no.
Floats omit the "E": `9.223500+4` = 9.2235e4.

- **MAT** — material id (U-235 = 9228).
- **MF** — file = data type. **MT** — section = reaction.
- **TAB1 record** → `endf.Tabulated1D` (x, y, breakpoints, interpolation laws).

| MF | Content | MF | Content |
|---|---|---|---|
| 1 | General info, ν̄ (MT452/455/456) | 6 | Energy–angle distributions |
| 2 | Resonance parameters (MT151) | 12–15 | Photon production |
| 3 | Cross sections σ(E) | 31–35 | Covariances (33 = σ covariances) |
| 4 | Angular distributions | 40 | Radionuclide production covariances |
| 5 | Energy distributions | | |

| MT | Reaction | MT | Reaction |
|---|---|---|---|
| 1 | total | 102 | (n,γ) radiative capture |
| 2 | elastic | 103 / 107 | (n,p) / (n,α) |
| 4 | inelastic (sum of 51–91) | 452 | ν̄ total |
| 16 / 17 | (n,2n) / (n,3n) | 455 / 456 | ν̄ delayed / prompt |
| 18 | fission (total) | 151 | resonance parameters (MF=2) |

Interpolation laws (INT): 1 histogram, 2 lin-lin, 3 lin-log (y linear in ln x), 4 log-lin,
5 log-log. Never resample with plain `np.interp` when INT≠2 — call the `Tabulated1D` object.

ENDF-6 manual: https://www.nndc.bnl.gov/endfdocs/ENDF-102-2023.pdf

---

## 6. `endf` Python package cheat sheet (v0.1.x, docs PDF v0.1.1)

```python
import endf

# Low-level
mat = endf.Material('data/raw/endf-b8.1/neutrons/n-092_U_235.endf')
mat.MAT, mat.sections                     # 9228, [(MF, MT), ...]
mat.section_text[12, 75]                  # raw 80-col text of a section
mat.section_data[3, 16]                   # dict: ZA, AWR, QM, QI, LR, sigma(Tabulated1D)
endf.get_materials(path)                  # list[Material] for multi-material files

# Tabulated1D (TAB1)
xs = mat.section_data[3, 16]['sigma']
xs.x, xs.y, xs.breakpoints, xs.interpolation, xs.n_pairs, xs.n_regions
xs(9.5e6); xs(np.linspace(6e6, 8e6, 10)); xs.integral()

# High-level
u235 = mat.interpret()                    # or endf.IncidentNeutron.from_endf(path)
u235.reactions                            # {MT: Reaction}
u235[18], u235['n,2n']                    # index by MT or reaction name
rx = u235[16]; rx.xs['0K']; rx.products; rx.q_reaction; rx.q_massdiff; rx.redundant
p = rx.products[-1]; p.name, p.yield_, p.emission_mode, p.distribution

# ACE (arrays are 1-based: index 0 is a padding zero)
table = endf.ace.get_table('data/raw/ace/80198.710nc')
table.nxs, table.jxs, table.xss, table.temperature, table.zaid, table.data_type
hg198 = table.interpret()                 # IncidentNeutron; xs keys like '294K'
endf.ace.get_tables(path), endf.ace.get_libraries_from_xsdir(xsdir)
```

Gotchas: ENDF-sourced cross sections only exist at `'0K'`; ACE data has temperature keys.
ACE energies are converted MeV→eV by `Tabulated1D.from_ace(convert_units=True)`.
Docs: https://endf-python.readthedocs.io

---

## 7. Data-engineering conventions (implemented in the thin pipeline)

Long/tidy Parquet partitioned by nuclide, one row per (nuclide, MT, temperature, energy):

```
za:int, nuclide:str, mat:int, mt:int, reaction:str, temperature_k:float,
energy_ev:double, xs_barn:double, interp_law:int, source:str ('endf'|'ace'|'point'), idx:int
```

`idx` is the point's position within its curve. ENDF repeats an energy at discontinuities, so
order by `idx`, never by `energy_ev`. `interp_law` is the law of the interval *ending* at that
point; `etl.arrays_to_tab` rebuilds an equivalent `Tabulated1D` from (energy, xs, interp_law).

Wide table (`xs_wide/`): one row per (nuclide, MT) with metadata + `n_points` + `xs`, an
`array<double>` of length N whose element j is σ at grid energy j. Grid definition is in
`xs_wide_grid.json` (emin, emax, n, energies).

---

## 8. Open questions / decisions log

Open:
- Dense pointwise source: POINT2025 vs ACE Lib81 vs reconstruct ourselves (openmc). *Undecided.*
  POINT2025 293.6 K was spot-checked: it parses directly with `endf.Material` and has 40–1,900×
  more points than MF=3 (e.g. U-238 MT102: 110 → 211,133). Leading candidate.
- UQ figure of merit: k∞ proxy vs real transport (OpenMC). *Undecided.*
- Cluster target: local Docker for dev; course cluster (Dataproc/HPC) for scale runs. *TBD.*
- Compressor: the masked AE beats PCA on reconstruction (below), but its nominal k∞ bias is
  larger (−0.061 vs PCA +0.038) because it underestimates the flux-weighted U-238 σf and σc.
  Candidates: a flux-importance-weighted loss, more epochs (best checkpoints sit at the end of
  training), and re-checking on dense POINT2025 data, where the curve structure differs.

Decisions (thin end-to-end pipeline, branch `feat/thin-e2e`, 2026-11-25):
- **Data**: thin version uses ENDF MF=3 at 0 K, MT 1/2/18/102 only. MF=3 is only a background
  in the RRR, e.g. U-235 MT18/102 are exactly 0 from 1e-5 eV to 2.24 keV. Absolute k∞ is
  therefore not physical; the pipeline is built so a pointwise source can replace stage 1 input.
- **ETL**: Spark parallelises over file paths; each task parses one file and emits one record
  per curve with arrays, which Spark explodes (avoids millions of Python Row objects). Failures
  are logged to `etl_errors.json`, not fatal. All 558 files currently parse.
- **Grid**: log-uniform, 1e-5 eV–20 MeV, N=2000 (quick: 500). Values outside a curve's range
  are 0. Wide rows store σ as an array column, not 2000 separate columns.
- **Features**: log10(σ + 1e-10). Zeros become −10, which creates cliffs PCA handles poorly.
- **PCA**: fitted once with up to 64 components; error = per-curve relative L2 in linear σ.
  `choose_k` = smallest k with median error ≤ 1 %, else the largest k. On the full run nothing
  meets 1 % (k=64 median 30 %), so k=64 is chosen.
- **Figure of merit**: one-group k∞ = Σ N ν̄σf / Σ N (σf + σc) for homogenised 4 % UO₂ + water
  (fuel/water volume 0.35/0.65, cold densities; derivation in `uq.py`). ν̄(E) is read from
  MF=1/MT=452 for U-235/U-238 (constants 2.43 / 2.8 are only a fallback). The reference flux is an
  analytic Maxwellian + 1/E + Watt mix (25/35/40 % of flux), not a transport solution.
- **MC**: both schemes are calibrated to ~0.75 % RMS relative σ perturbation and applied to the
  MT18/MT102 rows of U235/U238/H1/O16. Full space: squared-exponential correlation in log10 E,
  length 0.5 decades, factor via `eigh` (its cost is included in full-space timing). One Spark task
  per batch of 250 samples, seeded by (seed, method, batch) so results do not depend on
  partitioning. They are identical on local FS and HDFS. (Latent scheme: see the AE decisions below;
  PR #2 used an isotropic σ_z = 0.0075/ln10·√(N/k) instead.)

Decisions (autoencoder option, branch `feat/autoencoder`, 2026-11-26):
- **Option, not default**: `--compressor ae` adds the AE; PCA alone stays the default. PCA-only
  runs delete a stale `compressor_comparison.json` / `models/ae.pt` from the output dir.
- **Model**: MLP 2N → 512 → 256 → k → 256 → 512 → 2N. Input is [standardised log10 σ (0 where
  σ = 0) | presence mask]. Outputs are [mask logits | log-values]. Loss is BCE(mask) + MSE(log-value
  where σ > 0). Decoding gives exactly 0 where logit ≤ 0 and clips log-values to the training range.
  A 1-D conv AE was tried first and dropped: ~9 s/epoch on CPU vs ~1 s for the MLP.
- **Training**: AdamW + OneCycle (peak 2e-3), batch 64, 150 epochs. A validation check every 10
  epochs keeps the best checkpoint by held-out median error. The torch seed is set *before* the
  network is built; otherwise weight init depends on process RNG state and runs differ.
- **Fair comparison**: a shared 80/20 row split (seed 0). The reactor nuclides' rows are forced
  into training, because stage 3 encodes them. PCA is refit on the training split for the table,
  `compressor_comparison.json`. If no rows can be held out (e.g. `--nuclides` with only reactor
  nuclides), it falls back to in-sample evaluation and records `"in_sample": true`.
- **AE k**: `choose_k_tolerance` = smallest k meeting 1 %, else within 10 % of the best held-out
  median. "Largest k" would waste dimensions on a plateau. Full run → k=8.
- **Ray**: one Ray task per k (CPUs split evenly), run in a *spawned child process*. Starting Ray
  inside the Spark driver process breaks Py4J (ConnectionResetError on the next Spark call, even
  without `ray.shutdown()`).
- **Latent MC (all compressors)**: perturb z0 + c·s∘ε, where s is the per-dimension std of the
  compressor's codes over all rows. The scalar c is calibrated empirically (fixed-point on a fixed
  draw) to the target RMS relative σ perturbation, since AE latent units are arbitrary. This also
  changes the PCA latent MC: std 0.0018 (PR #2, isotropic) → 0.0043, now comparable to full space
  (0.0040).
- **Spark workers** decode the AE with `torch.set_num_threads(1)` to avoid oversubscription. The
  model is pickled via `torch.save` bytes (`__getstate__`).
- **HDFS**: `--hdfs` puts the Parquet datasets under `/heisenbig/<out-name>/`; JSON and model
  artifacts stay local. After `--profile hdfs up`, run
  `docker compose exec namenode hdfs dfsadmin -safemode wait` before writing, or the namenode
  rejects writes while still in safe mode.
