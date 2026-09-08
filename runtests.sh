#!/usr/bin/env bash
set -euo pipefail
ctest --preset "${1:-release}"
