@echo off
setlocal

set VCPKG_ROOT=%CD%\vcpkg
set BUILD_DIR=%CD%\build
set ROOT_DIR=%CD%

if not exist "%VCPKG_ROOT%" (
    echo Cloning vcpkg...
    git clone https://github.com/microsoft/vcpkg.git "%VCPKG_ROOT%"
    echo Bootstrapping vcpkg...
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
    ) else (
        echo Package %%P already installed
    )
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo Configuring CMake...
cmake -S . -B "%BUILD_DIR%" ^
    -G Ninja ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=%ROOT_DIR%

echo Building project...
cmake --build "%BUILD_DIR%" --config Release

echo Build complete!

endlocal
