@echo off
setlocal
echo Running tests...
if not exist build (
    echo Build directory not found. Please build the project first.
    exit /b 1
)

cd build

echo Running tests with CTest in Release mode...
ctest --output-on-failure -C Release

cd ..
endlocal