"""Fast tests for the thin end-to-end pipeline.

    docker compose exec dev pytest -q tests/test_pipeline.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import endf
from heisenbig import compress, etl, grid, uq

DATA = Path(os.environ.get("HEISENBIG_DATA", "data"))
RAW = DATA / "raw/endf-b8.1/neutrons"


# --- stage 1/2: interpolation-preserving storage and resampling --------------

def _two_region_tab():
    # region 1 (points 1-3) lin-lin, region 2 (points 3-5) log-log
    return endf.Tabulated1D([1, 10, 100, 1e3, 1e4], [1, 2, 3, 30, 300], breakpoints=[3, 5], interpolation=[2, 5])


def test_tab_arrays_round_trip_preserves_interpolation():
    tab = _two_region_tab()
    back = etl.arrays_to_tab(*etl.tab_to_arrays(tab))
    xs = np.logspace(0, 4, 97)
    np.testing.assert_allclose(back(xs), tab(xs))
    assert list(back.breakpoints) == [3, 5] and list(back.interpolation) == [2, 5]


def test_resample_respects_law_and_zero_outside():
    tab = endf.Tabulated1D([1, 10, 100], [1, 2, 3], breakpoints=[3], interpolation=[5])  # log-log
    e = np.array([0.5, 10 ** 0.5, 100, 200])
    out = grid.resample(tab, e)
    assert out[0] == 0 and out[-1] == 0                  # below / above tabulated range
    assert out[1] == pytest.approx(np.sqrt(2))           # log-log, np.interp would give 1.5
    assert out[2] == pytest.approx(3)


def test_nuclide_name():
    assert etl.nuclide_name("n-092_U_235.endf") == "U235"
    assert etl.nuclide_name("/x/n-095_Am_242m1.endf") == "Am242_m1"
    assert etl.nuclide_name("n-006_C_000.endf") == "C0"


def test_grid_save_load(tmp_path):
    g = grid.Grid(n=50)
    g.save(tmp_path / "g.json")
    assert grid.Grid.load(tmp_path / "g.json") == g
    assert len(json.loads((tmp_path / "g.json").read_text())["energies"]) == 50


# --- stage 2: compressor ------------------------------------------------------

def test_pca_encode_decode_round_trip():
    rng = np.random.default_rng(0)
    # exactly rank 3 in log space; values far above log10(EPS) = -10, so the clip never triggers
    feats = 0.3 * rng.normal(size=(40, 3)) @ rng.normal(size=(3, 120))
    assert feats.min() > -5
    xs = compress.from_features(feats)
    m = compress.PCACompressor(k=3, k_max=8).fit(xs)
    z = m.encode(xs)
    assert z.shape == (40, 3)
    np.testing.assert_allclose(m.decode(z), xs, rtol=1e-6)
    assert compress.relative_errors(xs, m.decode(z)).max() < 1e-6


def test_choose_k():
    rows = [{"k": 2, "median_rel_err": 0.5}, {"k": 4, "median_rel_err": 0.009}, {"k": 8, "median_rel_err": 0.001}]
    assert compress.choose_k(rows) == 4
    assert compress.choose_k(rows[:1]) == 2  # none meets target → largest


# --- stage 3: figure of merit -------------------------------------------------

def test_k_inf_toy_known_answer():
    e = np.logspace(-5, 7, 100)
    w = uq.collapse_weights(e, np.ones_like(e))
    one = np.ones_like(e)
    # single fissile nuclide: k = ν σf / (σf + σc) = 2.5 · 1 / 2 = 1.25
    assert uq.k_inf({"A": one}, {"A": one}, {"A": 2.5}, {"A": 1.0}, w) == pytest.approx(1.25)
    # add a pure absorber with N σc = 2 → k = 2.5 / (2 + 2) = 0.625
    k = uq.k_inf({"A": one}, {"A": one, "B": 2 * one}, {"A": 2.5}, {"A": 1.0, "B": 1.0}, w)
    assert k == pytest.approx(0.625)


def test_reference_flux_weights():
    e = grid.Grid(n=400).energies
    w = uq.collapse_weights(e, uq.reference_flux(e))
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)


def test_mc_batches_are_reproducible():
    e = grid.Grid(n=60).energies
    xs = np.vstack([np.full(60, 2.0), np.full(60, 1.0)])           # U235 MT18, MT102
    model = compress.PCACompressor(k=2, k_max=2).fit(np.vstack([xs, 1.5 * xs, 0.7 * xs]))
    ctx = {
        "rows": [("U235", 18), ("U235", 102)], "nu": {"U235": 2.4}, "densities": {"U235": 1.0},
        "w": uq.collapse_weights(e, np.ones_like(e)), "batch_size": 5, "n_samples": 12, "seed": 7,
        "z0": model.encode(xs), "components": model.components, "mean": model.mean,
        "xs_rec": model.decode(model.encode(xs)), "xs_orig": xs, "f0": compress.to_features(xs),
        "sigma_z": 0.01, "sigma_f": 0.003, "L": uq.corr_factor(np.log10(e), 0.5),
    }
    for method in ("latent", "full"):
        a, b = uq.run_batch(1, ctx, method), uq.run_batch(1, ctx, method)
        assert a == b and len(a) == 5 and a[0][1] == 5
        assert len(uq.run_batch(2, ctx, method)) == 2              # last, partial batch
        assert all(0.9 < r[2] / 1.6 < 1.1 for r in a)               # nominal k = 2.4·2/3 = 1.6


# --- end to end ---------------------------------------------------------------

@pytest.mark.skipif(not (RAW / "n-092_U_235.endf").exists(), reason="run scripts/download_endf.sh first")
def test_quick_end_to_end(tmp_path):
    from heisenbig import pipeline

    s = pipeline.run(str(RAW), str(tmp_path / "out"), nuclides=["U235", "U238", "H1", "O16"],
                     n_grid=200, n_samples=60, uq_cfg=uq.UQConfig(n_samples=60, batch_size=25))
    out = tmp_path / "out"
    for p in ("xs_long", "xs_wide", "uq/samples", "xs_wide_grid.json", "models/pca.joblib",
              "pca_errors.json", "uq_summary.json", "run_summary.json"):
        assert (out / p).exists(), p
    assert s["stages"]["1_etl"]["failed"] == 0 and s["stages"]["1_etl"]["nuclides"] == 4
    assert s["stages"]["2_grid_compress"]["matrix_shape"][1] == 200
    mc = s["stages"]["3_uq"]["mc"]
    assert all(np.isfinite(mc[m]["std"]) and mc[m]["std"] > 0 for m in ("latent", "full"))
