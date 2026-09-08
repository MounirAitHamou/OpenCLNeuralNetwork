#!/usr/bin/env bash
set -euo pipefail
mode="${1:-release}"
"out/build/$mode/clnn_examples"
