#!/usr/bin/env bash
# Usage: entrypoint.sh namenode|datanode
set -euo pipefail
role="${1:?role required: namenode|datanode}"
case "$role" in
  namenode)
    if [ ! -d /hadoop/dfs/name/current ]; then
      echo ">> Formatting HDFS namenode"
      hdfs namenode -format -force -nonInteractive
    fi
    exec hdfs namenode ;;
  datanode)
    exec hdfs datanode ;;
  *) exec "$@" ;;
esac
