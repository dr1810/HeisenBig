# Minimal single-node-capable HDFS image built from the official Hadoop binary tarball.
# (apache/hadoop on Docker Hub is amd64-only; this one is multi-arch via temurin.)
FROM eclipse-temurin:17-jre-jammy

ARG HADOOP_VERSION=3.4.2
ENV HADOOP_HOME=/opt/hadoop \
    HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop \
    PATH=/opt/hadoop/bin:/opt/hadoop/sbin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}-lean.tar.gz" \
        | tar -xz -C /opt \
    && mv /opt/hadoop-${HADOOP_VERSION} ${HADOOP_HOME} \
    && rm -rf ${HADOOP_HOME}/share/doc

COPY docker/hdfs/core-site.xml docker/hdfs/hdfs-site.xml ${HADOOP_CONF_DIR}/
COPY docker/hdfs/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /hadoop/dfs/name /hadoop/dfs/data

ENTRYPOINT ["/entrypoint.sh"]
