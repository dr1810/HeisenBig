#!/usr/bin/env bash
# Run the thin end-to-end pipeline (stages 1→3) inside the dev container.
#   bash scripts/run_thin_e2e.sh            # full: all ENDF files
#   bash scripts/run_thin_e2e.sh --quick    # a handful of nuclides, small N
# Any other flags are passed to `python -m heisenbig.pipeline` (see --help).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /.dockerenv ]; then
  exec python -m heisenbig.pipeline "$@"
else
  exec docker compose exec -T dev python -m heisenbig.pipeline "$@"
fi
