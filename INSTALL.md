# 🔧 Installation Guide – OpenCLNeuralNetwork (Windows)

This guide explains how to set up and build **OpenCLNeuralNetwork** on **Windows** using **Visual Studio 2022**, **prebuilt OpenCL binaries**, and **CMake**.

---

## 1. Install Required Tools

### ✅ C++ Compiler (MSVC)
- Install **Visual Studio 2022** (or Build Tools):  
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
- During installation:
  - Select **Desktop development with C++**
  - Enable **x64** support

### ✅ CMake
- Download and install CMake:  
  https://cmake.org/download/

---

## 2. Install OpenCL (Prebuilt – No Source Build)

This project uses the **official Khronos OpenCL SDK** binaries.

### 2.1. (Optional) Vulkan SDK
> ⚠️ Vulkan is **only required** if you plan to use OpenCL implementations that run  
> **on top of Vulkan** (e.g. `clvk`).  
> If you are using a **native OpenCL driver** (Intel / AMD / NVIDIA), Vulkan is **not required**.

- Download (optional):  
  https://vulkan.lunarg.com/sdk/home

---

### 2.2. Download OpenCL SDK (Release)

- Go to the OpenCL SDK releases page:  
  https://github.com/KhronosGroup/OpenCL-SDK/releases
- Download the **Windows prebuilt SDK** (ZIP or installer)
- Extract/install it to a location of your choice, e.g.:
  `C:\OpenCL-SDK`

This provides:
- OpenCL headers
- OpenCL ICD loader (`OpenCL.lib`)
- CMake configuration files

> 💡 You still need a **vendor OpenCL runtime** (GPU or CPU driver):
> - NVIDIA: comes with GPU driver
> - AMD: Adrenalin / ROCm
> - Intel: oneAPI or Intel OpenCL runtime

## 3. Install CLBlast (Prebuilt Release)

This project uses **prebuilt CLBlast binaries** (no source build required).

- Go to the CLBlast releases page:  
https://github.com/CNugteren/CLBlast/releases
- Download the **Windows prebuilt release**
- Extract it to a directory of your choice, for example:
  `C:\CLBlast`

This provides:
- CLBlast headers
- CLBlast library (`clblast.lib` and `clblast.dll`)

> ⚠️ The runtime DLL (`clblast.dll`) must be available at runtime.  
> This project’s CMake configuration automatically copies it next to the executable.

## 4. Install HDF5 Library
- Download the HDF5 library from the official HDF Group releases page:
  - Visit https://github.com/HDFGroup/hdf5/releases/tag/hdf5_1.14.6
- Download hdf5-1.14.6-win-vs2022_cl.msi
- Install the HDF5 library using the installer.
- This will install the HDF5 library to a default location, typically `C:\Program Files\HDF_Group\HDF5\1.14.6`.
- If you installed it to a different location, make sure to specify the correct path in the CMake configuration step below.

## 5. Configure Project

Open `CMakeLists.txt` in the root of **OpenCLNeuralNetwork** and set the paths if needed.

## 6. Build the Project
- From the root of the OpenCLNeuralNetwork project:
```bash
build.bat
```

## 7. Make sure the tests pass
- After building, you can run the tests to ensure everything is working correctly:
```bash
runtests.bat
```

## 8. Run the Program
- After a successful build, you can run the program:
```bash
OpenCLNeuralNetwork.exe
```
- The program will execute the main function defined in `src/main.cpp`, which includes a simple neural network training example.

✅ Done
You're now ready to experiment with the OpenCL-accelerated neural network!