@echo off
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=release
call setup.bat %MODE% || exit /b 1
ctest --preset %MODE% || exit /b 1
