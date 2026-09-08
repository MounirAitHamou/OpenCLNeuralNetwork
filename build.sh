#!/usr/bin/env bash
set -euo pipefail
mode="${1:-release}"
"$(dirname "$0")/setup.sh" "$mode"
ctest --preset "$mode"
