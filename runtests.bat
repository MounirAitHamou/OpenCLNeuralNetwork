@echo off
setlocal

set MODE=%1

if "%MODE%"=="" set MODE=release

if /I not "%MODE%"=="debug" if /I not "%MODE%"=="release" (
    echo Usage: runtests.bat [debug^|release]
    exit /b 1
)

echo Running tests in %MODE% mode with CTest...

set BUILD_DIR=build-%MODE%

if not exist "%BUILD_DIR%" (
    echo Build directory "%BUILD_DIR%" not found. Please build the project first.
    exit /b 1
)

cd "%BUILD_DIR%"

if /I "%MODE%"=="debug" (
    ctest --output-on-failure -C Debug
) else (
    ctest --output-on-failure -C Release
)

cd ..
endlocal