@echo off
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=release
cmake --preset %MODE% || exit /b 1
cmake --build --preset %MODE% || exit /b 1
