@echo off
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=release
out\build\%MODE%\clnn_examples.exe
