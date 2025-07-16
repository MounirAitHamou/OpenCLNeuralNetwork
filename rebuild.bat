@echo off
setlocal

if exist build (
    rmdir /s /q build
)
mkdir build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release

ninja

cd ..
endlocal