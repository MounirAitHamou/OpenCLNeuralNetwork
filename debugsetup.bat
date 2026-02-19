@echo off
setlocal

set VCPKG_ROOT=%CD%\vcpkg
set BUILD_DIR=%CD%\build-debug



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

echo Configuring Debug + AddressSanitizer build...

cmake -S . -B "%BUILD_DIR%" ^
    -G Ninja ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_CXX_FLAGS="/fsanitize=address /Zi /RTC1 /W4 /permissive-" ^
    -DCMAKE_EXE_LINKER_FLAGS="/fsanitize=address"

echo Building Debug version...
cmake --build "%BUILD_DIR%"


endlocal
