@echo off
setlocal

set MODE=%1
if "%MODE%"=="" set MODE=release

if /I not "%MODE%"=="debug" if /I not "%MODE%"=="release" (
    echo Usage: tidy.bat [debug^|release]
    exit /b 1
)

if /I "%MODE%"=="debug" (
    set BUILD_DIR=build-debug
) else (
    set BUILD_DIR=build-release
)

if not exist "%BUILD_DIR%\compile_commands.json" (
    echo Error: %BUILD_DIR%\compile_commands.json not found.
    echo Please run CMake for the %MODE% configuration first.
    exit /b 1
)

echo Running Clang-Tidy against %BUILD_DIR%...

python "C:\Program Files\LLVM\bin\run-clang-tidy" ^
    -p %BUILD_DIR% ^
    -checks="modernize-*,readability-*,performance-*,-modernize-use-trailing-return-type,-readability-magic-numbers,-readability-function-cognitive-complexity,-readability-identifier-length" ^
    -header-filter="^(include/|src/)"

endlocal