@echo off
setlocal

set MODE=%1

if "%MODE%"=="" set MODE=release

if /I not "%MODE%"=="debug" if /I not "%MODE%"=="release" (
    echo Usage: run.bat [debug^|release]
    exit /b 1
)

if /I "%MODE%"=="debug" (
    set EXE_PATH=build-debug\OpenCLNeuralNetwork.exe
) else (
    set EXE_PATH=OpenCLNeuralNetwork.exe
)

if not exist "%EXE_PATH%" (
    echo Executable "%EXE_PATH%" not found. Please build the project first.
    exit /b 1
)

echo Running %EXE_PATH%...
"%EXE_PATH%"

endlocal