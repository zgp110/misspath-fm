#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

cd "$(dirname "$0")"

bash run_ettm1.sh       "${GPU}"
bash run_synthetic.sh   "${GPU}"
bash run_electricity.sh "${GPU}"
bash run_physionet.sh   "${GPU}"
