set -e

echo "Running tests..."

BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Please build the project first."
    exit 1
fi

cd "$BUILD_DIR"

ctest --output-on-failure -C Release

cd ..
echo "Tests finished."
