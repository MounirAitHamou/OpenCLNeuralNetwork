@echo off
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=release
ctest --preset %MODE%
