"""Stage 1 — ETL: ENDF-6 files → long-format Parquet (AGENTS.md §7 schema).

Spark distributes the *file paths*; each worker parses its own file with the
``endf`` package and emits one record per (file, MT) curve with the points as
arrays. Spark then explodes the arrays into one row per energy point. The driver
never parses a material.

    python -m heisenbig.etl --nuclides U235 U238 H1 O16
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# MF=3 reactions kept by the thin pipeline.
REACTIONS = {1: "total", 2: "elastic", 18: "fission", 102: "capture"}

_FNAME = re.compile(r"^n-(\d{3})_([A-Za-z]+)_(\d{3})(m\d+)?\.endf$")


def nuclide_name(path: str | Path) -> str:
    """GNDS-style name from an NNDC filename: n-092_U_235.endf → 'U235',
    n-095_Am_242m1.endf → 'Am242_m1', n-006_C_000.endf → 'C0' (natural element)."""
    m = _FNAME.match(Path(path).name)
    if not m:
        raise ValueError(f"unrecognised ENDF filename: {path}")
    _, sym, a, meta = m.groups()
    return f"{sym}{int(a)}" + (f"_{meta}" if meta else "")


def list_endf_files(raw_dir: str | Path, nuclides: list[str] | None = None) -> list[str]:
    """All *.endf files in ``raw_dir`` (sorted), optionally limited to ``nuclides``."""
    files = sorted(str(p) for p in Path(raw_dir).glob("*.endf"))
    if nuclides:
        wanted = set(nuclides)
        files = [f for f in files if nuclide_name(f) in wanted]
        missing = wanted - {nuclide_name(f) for f in files}
        if missing:
            raise FileNotFoundError(f"no ENDF file for: {sorted(missing)} in {raw_dir}")
    return files


# ---------------------------------------------------------------------------
# Tabulated1D <-> per-point arrays
#
# ENDF NBT/INT: region r covers 1-based points NBT[r-1]..NBT[r]; the interval
# ending at 1-based point p (p > NBT[r-1], p <= NBT[r]) uses INT[r]. We store, per
# point, the law of the interval that ENDS at that point (the first point copies
# the second), which is enough to rebuild an equivalent Tabulated1D.
# ---------------------------------------------------------------------------

def tab_to_arrays(tab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(tab.x, dtype=float)
    y = np.asarray(tab.y, dtype=float)
    nbt = np.asarray(tab.breakpoints, dtype=int)
    laws = np.asarray(tab.interpolation, dtype=int)
    counts = np.diff(np.concatenate([[0], nbt]))
    law = np.repeat(laws, counts)
    if law.size != x.size:  # defensive: malformed NBT → single region
        law = np.full(x.size, laws[-1] if laws.size else 2)
    return x, y, law.astype(np.int32)


def arrays_to_tab(x, y, law):
    import endf

    x = np.asarray(x, dtype=float)
    law = np.asarray(law, dtype=int).copy()
    if law.size > 1:
        law[0] = law[1]
    # Each run of equal laws is one region; it ends (1-based NBT) at the run's last point.
    run_ends = np.nonzero(np.diff(law))[0]          # 0-based last index of each run but the final one
    nbt = [int(i) + 1 for i in run_ends] + [len(x)]
    ints = [int(law[n - 1]) for n in nbt]
    return endf.Tabulated1D(x, np.asarray(y, dtype=float), nbt, ints)


def parse_endf_file(path: str) -> list[tuple]:
    """Worker-side: parse one file → [('ok', record) | ('err', path, message)]."""
    import warnings

    try:
        import endf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "MF=31 ... ignored" noise
            mat = endf.Material(path)
        name = nuclide_name(path)
        out = []
        for mt, reaction in REACTIONS.items():
            sec = mat.section_data.get((3, mt))
            if sec is None:
                continue
            x, y, law = tab_to_arrays(sec["sigma"])
            out.append(("ok", {
                "za": int(sec["ZA"]), "nuclide": name, "mat": int(mat.MAT), "mt": mt,
                "reaction": reaction, "temperature_k": 0.0,
                "energy_ev": x.tolist(), "xs_barn": y.tolist(), "interp_law": law.tolist(),
            }))
        if not out:
            out.append(("err", path, "no MF=3 section among MT " + str(list(REACTIONS))))
        return out
    except Exception as exc:  # skip + log, never crash the job
        return [("err", path, f"{type(exc).__name__}: {exc}")]


def run_etl(spark, raw_dir: str | Path, out_uri: str, nuclides: list[str] | None = None,
            error_log: str | Path | None = None) -> dict:
    """Parse ENDF files in parallel and write long Parquet partitioned by nuclide."""
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    files = list_endf_files(raw_dir, nuclides)
    log.info("ETL: %d ENDF files", len(files))
    sc = spark.sparkContext
    parsed = sc.parallelize(files, numSlices=max(1, len(files))).flatMap(parse_endf_file)
    parsed.persist(StorageLevel.MEMORY_AND_DISK)

    errors = parsed.filter(lambda r: r[0] == "err").map(lambda r: (r[1], r[2])).collect()
    for path, msg in errors:
        log.warning("ETL skipped %s: %s", path, msg)

    curve_schema = T.StructType([
        T.StructField("za", T.IntegerType()), T.StructField("nuclide", T.StringType()),
        T.StructField("mat", T.IntegerType()), T.StructField("mt", T.IntegerType()),
        T.StructField("reaction", T.StringType()), T.StructField("temperature_k", T.DoubleType()),
        T.StructField("energy_ev", T.ArrayType(T.DoubleType())),
        T.StructField("xs_barn", T.ArrayType(T.DoubleType())),
        T.StructField("interp_law", T.ArrayType(T.IntegerType())),
    ])
    curves = spark.createDataFrame(parsed.filter(lambda r: r[0] == "ok").map(lambda r: r[1]), curve_schema)

    long_df = (
        curves.select(
            "za", "nuclide", "mat", "mt", "reaction", "temperature_k",
            F.posexplode(F.arrays_zip("energy_ev", "xs_barn", "interp_law")).alias("idx", "p"),
        )
        .select(
            "za", "nuclide", "mat", "mt", "reaction", "temperature_k",
            F.col("p.energy_ev").alias("energy_ev"), F.col("p.xs_barn").alias("xs_barn"),
            F.col("p.interp_law").alias("interp_law"),
            F.lit("endf").alias("source"),
            # point order within a curve; needed because ENDF repeats energies at
            # discontinuities, so sorting by energy alone is ambiguous.
            F.col("idx").cast("int").alias("idx"),
        )
    )
    long_df.write.mode("overwrite").partitionBy("nuclide").parquet(out_uri)
    parsed.unpersist()

    written = spark.read.parquet(out_uri)
    stats = {
        "files": len(files), "failed": len(errors),
        "rows": written.count(),
        "curves": written.select("nuclide", "mt").distinct().count(),
        "nuclides": written.select("nuclide").distinct().count(),
    }
    if error_log:
        Path(error_log).parent.mkdir(parents=True, exist_ok=True)
        Path(error_log).write_text(json.dumps([{"path": p, "error": m} for p, m in errors], indent=1))
    log.info("ETL done: %s", stats)
    return stats


def main(argv=None):
    from heisenbig.spark import get_spark

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw/endf-b8.1/neutrons")
    ap.add_argument("--out", default="data/processed/xs_long")
    ap.add_argument("--nuclides", nargs="*", help="limit to these nuclides, e.g. U235 H1 (default: all)")
    ap.add_argument("--hdfs", action="store_true", help="write to HDFS (out is then an HDFS path)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    spark = get_spark("heisenbig-etl", hdfs=a.hdfs)
    print(run_etl(spark, a.raw_dir, a.out, a.nuclides))


if __name__ == "__main__":
    main()
