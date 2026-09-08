#!/usr/bin/env bash
set -euo pipefail
mode="${1:-debug}"
run-clang-tidy -p "out/build/$mode"
