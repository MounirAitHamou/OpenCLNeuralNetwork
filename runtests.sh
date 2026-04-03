#!/usr/bin/env bash
set -e

MODE=$1

if [ -z "$MODE" ]; then
    MODE="release"
fi

if [ "$MODE" != "debug" ] && [ "$MODE" != "release" ]; then
    echo "Usage: ./runtests.sh [debug|release]"
    exit 1
fi

echo "Running tests in $MODE mode with CTest..."

BUILD_DIR="build-$MODE"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory \"$BUILD_DIR\" not found. Please build the project first."
    exit 1
fi

cd "$BUILD_DIR"

if [ "$MODE" = "debug" ]; then
    ctest --output-on-failure -C Debug
else
    ctest --output-on-failure -C Release
fi

cd -