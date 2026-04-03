# 🔧 Installation Guide – OpenCLNeuralNetwork

This guide explains how to build and run **OpenCLNeuralNetwork** on **Windows** and **Linux** using **CMake**, **vcpkg**, and a system OpenCL runtime.

---

## 1. Prerequisites

### 🪟 Windows

#### ✅ C++ Compiler (MSVC)
- Install **Visual Studio 2022** (or Build Tools):  
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
- During installation, select:
  - **Desktop development with C++**
  - **x64** toolchain

#### ✅ CMake
- Download and install CMake:  
  https://cmake.org/download/

#### ✅ OpenCL Runtime (Required)
> ⚠️ **Important:** vcpkg provides OpenCL headers and loaders, but **not** a working OpenCL device.

You must install an OpenCL runtime from your hardware vendor:
- **NVIDIA GPU** → NVIDIA Graphics Driver  
- **AMD GPU** → AMD Adrenalin Driver  
- **Intel GPU / CPU** → Intel OpenCL Runtime  

You can verify your installation later using tools like `clinfo`.

---

### 🐧 Linux (Ubuntu / Debian)

Install build tools, CMake, and OpenCL dependencies:
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git \
                    ocl-icd-opencl-dev clinfo pocl-opencl-icd
```

---

## 2. Build the Project
- Navigate to the root of the OpenCLNeuralNetwork repository.
- Run the platform-specific build script:
```bash
build.bat # Windows
./build.sh # Linux
```

What the build script does:
- Clones and sets up **vcpkg** if not already present.
- Installs required dependencies:
  - OpenCL headers and loader (via vcpkg)
  - HDF5 (for data storage)
  - CLBlast (for optimized OpenCL BLAS operations)
- Configures and builds the project using CMake and Ninja.
- Runs unit tests to verify the build.

---

## 3. Run the Program
- After a successful build, execute the program:
```bash
run.bat # Windows
./run.sh # Linux
```

- The program will run the main function defined in `src/main.cpp`, which includes a simple neural network training example.

---

## 4.Troubleshooting
- No OpenCL platforms found
  - Ensure you have installed the correct OpenCL runtime for your hardware.
  - Verify installation with `clinfo` to see available platforms and devices.
- Build succeeds but runtime crashes
  - Verify that your graphics drivers are up to date.
  - Ensure the OpenCL runtime matches your hardware (e.g., NVIDIA drivers for NVIDIA GPUs).

---

## ✅ Done
You're now ready to experiment with the OpenCL-accelerated neural network!
