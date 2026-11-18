"""Smoke tests: every teammate's container should pass these.

    docker compose exec dev pytest -q
"""
import os
from pathlib import Path

import numpy as np
import pytest

DATA = Path(os.environ.get("HEISENBIG_DATA", "data"))
U235 = DATA / "raw/endf-b8.1/neutrons/n-092_U_235.endf"


def test_endf_tabulated1d():
    import endf

    f = endf.Tabulated1D([0, 10], [4, 5])
    assert [f(x) for x in np.linspace(0, 10, 5)] == [4.0, 4.25, 4.5, 4.75, 5.0]


@pytest.mark.skipif(not U235.exists(), reason="run scripts/download_endf.sh first")
def test_endf_reads_u235():
    import endf

    u235 = endf.IncidentNeutron.from_endf(U235)
    assert u235.atomic_number == 92 and u235.mass_number == 235
    assert 18 in u235.reactions


def test_spark_local():
    from heisenbig.spark import get_spark

    spark = get_spark("smoke")
    df = spark.range(1000).selectExpr("id", "log(id + 1) AS y")
    assert df.count() == 1000
    pdf = df.limit(5).toPandas()  # exercises Arrow path
    assert len(pdf) == 5


def test_torch_cpu():
    import torch

    x = torch.randn(64, 32)
    model = torch.nn.Sequential(torch.nn.Linear(32, 4), torch.nn.Linear(4, 32))
    loss = torch.nn.functional.mse_loss(model(x), x)
    loss.backward()
    assert torch.isfinite(loss)


def test_ray_local():
    import ray

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        @ray.remote
        def sq(v):
            return v * v

        assert sum(ray.get([sq.remote(i) for i in range(10)])) == 285
    finally:
        ray.shutdown()
