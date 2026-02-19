#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Variables
# -------------------------------
VCPKG_ROOT="$PWD/vcpkg"
BUILD_DIR="$PWD/build"
PACKAGES=("opencl:x64-linux" "hdf5[cpp,hl]:x64-linux" "clblast:x64-linux")

# -------------------------------
# Function: Check OpenCL devices
# -------------------------------
check_opencl_available() {
    if command -v clinfo &> /dev/null; then
        echo "OpenCL detected (first platforms/devices):"
        # ignore clinfo exit code
        clinfo | grep -E "Platform Name|Device Name" | head -n 10 || true
    else
        echo "Warning: clinfo not found. Your project may not run correctly without OpenCL runtime."
    fi
}

# -------------------------------
# Function: Setup vcpkg
# -------------------------------
setup_vcpkg() {
    if [ ! -d "$VCPKG_ROOT" ]; then
        echo "Cloning vcpkg..."
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
        echo "Bootstrapping vcpkg..."
        "$VCPKG_ROOT/bootstrap-vcpkg.sh"
    else
        echo "vcpkg already exists, skipping clone"
    fi

    for pkg in "${PACKAGES[@]}"; do
        if ! "$VCPKG_ROOT/vcpkg" list | grep -iq "$pkg"; then
            echo "Installing $pkg..."
            "$VCPKG_ROOT/vcpkg" install "$pkg"
        else
            echo "Package $pkg already installed, skipping"
        fi
    done
}

# -------------------------------
# Function: Build project
# -------------------------------
build_project() {
    mkdir -p "$BUILD_DIR"
    echo "Configuring CMake..."
    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
        -DCMAKE_BUILD_TYPE=Release

    echo "Building project..."
    cmake --build "$BUILD_DIR" --config Release
    echo "Build complete!"
}

# -------------------------------
# Main
# -------------------------------
echo "=== Checking OpenCL runtime ==="
check_opencl_available

echo "=== Setting up vcpkg ==="
setup_vcpkg

echo "=== Building project ==="
build_project
