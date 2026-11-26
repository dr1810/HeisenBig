"""Stage 2a — resample every (nuclide, MT) curve onto one shared log-uniform grid.

Evaluation goes through ``endf.Tabulated1D`` so the ENDF interpolation laws are
respected (never plain ``np.interp``, AGENTS.md §5); values outside a curve's
tabulated range (e.g. below a threshold) are 0.

Output: a "wide" Parquet table, one row per (nuclide, MT), with the resampled
values in an array column ``xs`` of length N (column j ↔ grid energy j), and the
grid definition as JSON next to it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from heisenbig.etl import arrays_to_tab


@dataclass(frozen=True)
class Grid:
    emin: float = 1e-5   # eV
    emax: float = 2e7    # eV (20 MeV)
    n: int = 2000

    @property
    def energies(self) -> np.ndarray:
        return np.logspace(np.log10(self.emin), np.log10(self.emax), self.n)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self) | {"spacing": "log-uniform", "unit": "eV", "energies": self.energies.tolist()}
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path: str | Path) -> "Grid":
        d = json.loads(Path(path).read_text())
        return cls(d["emin"], d["emax"], d["n"])


def resample(tab, energies: np.ndarray) -> np.ndarray:
    """Evaluate a Tabulated1D on ``energies``; exactly 0 outside [x0, x_last]."""
    energies = np.asarray(energies, dtype=float)
    out = np.zeros_like(energies)
    x = np.asarray(tab.x)
    inside = (energies >= x[0]) & (energies <= x[-1])
    if inside.any():
        out[inside] = np.asarray(tab(energies[inside]), dtype=float)
    return out


def resample_long(spark, long_uri: str, wide_uri: str, grid: Grid, grid_path: str | Path) -> dict:
    """Group the long table by curve and resample each group with applyInPandas."""
    import pandas as pd
    from pyspark.sql import types as T

    energies = grid.energies
    key_cols = ["nuclide", "za", "mat", "mt", "reaction", "temperature_k"]

    def _resample_group(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("idx")
        tab = arrays_to_tab(pdf["energy_ev"].to_numpy(), pdf["xs_barn"].to_numpy(),
                            pdf["interp_law"].to_numpy())
        row = {c: pdf[c].iloc[0] for c in key_cols}
        row["n_points"] = len(pdf)
        row["xs"] = resample(tab, energies).tolist()
        return pd.DataFrame([row])

    schema = T.StructType([
        T.StructField("nuclide", T.StringType()), T.StructField("za", T.IntegerType()),
        T.StructField("mat", T.IntegerType()), T.StructField("mt", T.IntegerType()),
        T.StructField("reaction", T.StringType()), T.StructField("temperature_k", T.DoubleType()),
        T.StructField("n_points", T.IntegerType()),
        T.StructField("xs", T.ArrayType(T.DoubleType())),
    ])
    long_df = spark.read.parquet(long_uri)
    wide = long_df.groupBy("nuclide", "mt").applyInPandas(_resample_group, schema)
    wide.orderBy("nuclide", "mt").write.mode("overwrite").parquet(wide_uri)
    grid.save(grid_path)
    n_rows = spark.read.parquet(wide_uri).count()
    return {"rows": n_rows, "grid_n": grid.n, "emin": grid.emin, "emax": grid.emax}


def load_wide(spark, wide_uri: str):
    """Wide table → (pandas metadata frame, X[rows, N] numpy matrix)."""
    pdf = spark.read.parquet(wide_uri).orderBy("nuclide", "mt").toPandas()
    X = np.vstack(pdf["xs"].to_numpy()) if len(pdf) else np.zeros((0, 0))
    return pdf.drop(columns=["xs"]), X
