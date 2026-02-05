@echo off
setlocal

if not exist build (
    mkdir build
)

cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release

ninja

cd ..

runtests.bat

endlocal
