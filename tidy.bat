@echo off
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=debug
run-clang-tidy -p out\build\%MODE%
