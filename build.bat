@echo off
setlocal

set MODE=%1

if "%MODE%"=="" (
    set MODE=release
)

call setup.bat %MODE%
call runtests.bat %MODE%

endlocal