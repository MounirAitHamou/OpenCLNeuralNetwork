#!/usr/bin/env bash
set -e

MODE=$1

if [ -z "$MODE" ]; then
    MODE="release"
fi

if [ "$MODE" != "debug" ] && [ "$MODE" != "release" ]; then
    echo "Usage: ./run.sh [debug|release]"
    exit 1
fi

if [ "$MODE" = "debug" ]; then
    EXE_PATH="build-debug/OpenCLNeuralNetwork"
else
    EXE_PATH="OpenCLNeuralNetwork"
fi

if [ ! -f "$EXE_PATH" ]; then
    echo "Executable \"$EXE_PATH\" not found. Please build the project first."
    exit 1
fi

echo "Running $EXE_PATH..."
"$EXE_PATH"