#!/usr/bin/env bash
set -euo pipefail

mode="${1:-release}"
if [[ "$mode" != "debug" && "$mode" != "release" ]]; then
    echo "usage: ./setup.sh [debug|release]" >&2
    exit 2
fi
cmake --preset "$mode"
cmake --build --preset "$mode"
