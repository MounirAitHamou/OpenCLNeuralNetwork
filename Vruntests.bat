@echo off
setlocal

if not exist build (
    echo Build directory not found. Please build the project first.
    exit /b 1
)

cd build

ctest -V --no-compress-output -C Release

cd ..
endlocal