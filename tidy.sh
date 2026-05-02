#!/usr/bin/env bash
set -e

MODE="$1"
if [ -z "$MODE" ]; then
    MODE="release"
fi

if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
    echo "Usage: ./tidy.sh [debug|release]"
    exit 1
fi

if [[ "$MODE" == "debug" ]]; then
    BUILD_DIR="build-debug"
else
    BUILD_DIR="build-release"
fi

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
    echo "Error: $BUILD_DIR/compile_commands.json not found."
    echo "Please run CMake for the $MODE configuration first."
    exit 1
fi

echo "Running Clang-Tidy against $BUILD_DIR..."

python3 /usr/bin/run-clang-tidy \
    -p "$BUILD_DIR" \
    -checks="modernize-*,readability-*,performance-*,-modernize-use-trailing-return-type,-readability-magic-numbers,-readability-function-cognitive-complexity,-readability-identifier-length" \
    -header-filter="^(include/|src/)"