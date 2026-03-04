#!/usr/bin/env bash
set -e

MODE=$1

if [ -z "$MODE" ]; then
    MODE=release
fi

if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
    echo "Usage: ./build.sh [debug|release]"
    exit 1
fi

./setup.sh "$MODE"

./runtests.sh "$MODE"