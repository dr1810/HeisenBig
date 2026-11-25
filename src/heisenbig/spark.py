"""SparkSession factory shared by notebooks, scripts and tests."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "heisenbig", hdfs: bool = False, **conf: str) -> SparkSession:
    """Return a local-mode SparkSession.

    Parameters
    ----------
    app_name : Spark application name.
    hdfs : if True, default filesystem is the compose HDFS namenode
        (start it with ``docker compose --profile hdfs up -d``).
    conf : extra Spark config, e.g. ``**{"spark.sql.shuffle.partitions": "64"}``.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if hdfs:
        builder = (
            builder.config("spark.hadoop.fs.defaultFS", os.environ.get("HDFS_URL", "hdfs://namenode:9000"))
            # Datanode is reached by container hostname, not its internal IP.
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        )
    for k, v in conf.items():
        builder = builder.config(k, v)
    return builder.getOrCreate()
