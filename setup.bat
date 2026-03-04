@echo off
setlocal

set MODE=%1

if "%MODE%"=="" (
    set MODE=release
)

if /I not "%MODE%"=="debug" if /I not "%MODE%"=="release" (
    echo Usage: setup.bat [debug^|release]
    exit /b 1
)

set ROOT_DIR=%CD%
set VCPKG_ROOT=%ROOT_DIR%\vcpkg
set BUILD_DIR=%ROOT_DIR%\build-%MODE%

if not exist "%VCPKG_ROOT%" (
    echo Cloning vcpkg...
    git clone https://github.com/microsoft/vcpkg.git "%VCPKG_ROOT%"
    call "%VCPKG_ROOT%\bootstrap-vcpkg.bat"
) else (
    echo vcpkg already exists
)

set PACKAGES="opencl:x64-windows" "hdf5[cpp,hl]:x64-windows" "clblast:x64-windows"

for %%P in (%PACKAGES%) do (
    "%VCPKG_ROOT%\vcpkg.exe" list | findstr /i "%%P" >nul
    if errorlevel 1 (
        echo Installing %%P...
        call "%VCPKG_ROOT%\vcpkg.exe" install %%P
    )
)

if /I "%MODE%"=="debug" (
    set BUILD_TYPE=Debug
    set OUTPUT_DIR=%BUILD_DIR%
) else (
    set BUILD_TYPE=Release
    set OUTPUT_DIR=%ROOT_DIR%
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo Configuring %BUILD_TYPE% build...

cmake -S . -B "%BUILD_DIR%" ^
    -G Ninja ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%OUTPUT_DIR%

cmake --build "%BUILD_DIR%"

echo Build complete!

endlocal