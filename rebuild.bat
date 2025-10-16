@echo off
setlocal

if exist build (
    rmdir /s /q build
)
if exist OpenCLNeuralNetwork.exe (
    del OpenCLNeuralNetwork.exe
)
mkdir build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release

ninja

cd ..
endlocal