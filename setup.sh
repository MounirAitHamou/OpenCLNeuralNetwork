#!/usr/bin/env bash

set -e

MODE=$1

if [ -z "$MODE" ]; then
    MODE=release
fi

if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
    echo "Usage: ./setup.sh [debug|release]"
    exit 1
fi

ROOT_DIR="$(pwd)"
VCPKG_ROOT="$ROOT_DIR/vcpkg"
BUILD_DIR="$ROOT_DIR/build-$MODE"

if [ ! -d "$VCPKG_ROOT" ]; then
    echo "Cloning vcpkg..."
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh"
else
    echo "vcpkg already exists"
fi

PACKAGES=(
    opencl
    hdf5[cpp,hl]
    clblast
)

for PKG in "${PACKAGES[@]}"; do
    if ! "$VCPKG_ROOT/vcpkg" list | grep -i "$PKG" > /dev/null; then
        echo "Installing $PKG..."
        "$VCPKG_ROOT/vcpkg" install "$PKG"
    fi
done

mkdir -p "$BUILD_DIR"

if [ "$MODE" = "debug" ]; then
    BUILD_TYPE=Debug
    OUTPUT_DIR="$BUILD_DIR"
else
    BUILD_TYPE=Release
    OUTPUT_DIR="$ROOT_DIR"
fi

echo "Configuring $BUILD_TYPE build..."

cmake -S . -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$OUTPUT_DIR"

cmake --build "$BUILD_DIR"

echo "Build complete!"