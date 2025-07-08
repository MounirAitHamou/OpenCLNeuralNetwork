# 🔧 Installation Guide – OpenCLNeuralNetwork

This guide will help you set up and build the **OpenCLNeuralNetwork** project on Windows using Visual Studio 2022 and OpenCL.

---

## 1. Install Required Tools

### ✅ C++ Compiler (MSVC)
- Download and install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- During installation, make sure to:
  - Select the **"Desktop development with C++"** workload.
  - Enable **x64 architecture** support.

### ✅ CMake
- Download and install CMake from:  
  [https://cmake.org/download/](https://cmake.org/download/)

---

## 2. Build the OpenCL SDK

This project uses the official OpenCL SDK provided by Khronos.

### Step-by-step:

#### 2.1. Install Vulkan SDK (Required by Intel OpenCL SDK)
- Go to [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home)
- Download and install the Vulkan SDK appropriate for your system.

#### 2.2. Clone the OpenCL SDK

git clone https://github.com/KhronosGroup/OpenCL-SDK.git

#### 2.3. Build the OpenCL SDK
- Open a terminal and navigate to the parent directory of OpenCL-SDK (e.g. C:/GitHub for me, I pulled it to C:/GitHub/OpenCL-SDK).
- Run the following commands:
```bash
cmake -A x64 -D CMAKE_INSTALL_PREFIX=./OpenCL-SDK/install -B ./OpenCL-SDK/build -S ./OpenCL-SDK
cmake --build OpenCL-SDK/build --config Release --target install -- /m /v:minimal
```

#### 3. Configure Project

- In the root directory of OpenCLNeuralNetwork, open CMakeLists.txt and set your OpenCL SDK path:
```cmake
set(OPENCL_SDK_PATH "C:/GitHub/OpenCL-SDK/install")
```


#### 4. Build the Project
- From the root of the OpenCLNeuralNetwork project:
```bash
rebuild.bat
```

#### 5. Run the Program
- After a successful build, you can run the program:
```bash
OpenCLNeuralNetwork.exe
```
- The program will execute the main function defined in `src/main.cpp`, which includes a simple neural network training example.

✅ Done
You're now ready to experiment with the OpenCL-accelerated neural network!




