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

#### 2.2. Clone and Build the OpenCL SDK
- git clone https://github.com/KhronosGroup/OpenCL-SDK.git
- Open a terminal and navigate to the parent directory of OpenCL-SDK (e.g. C:/GitHub for me, I pulled it to C:/GitHub/OpenCL-SDK).
- Run the following commands:
```bash
cmake -A x64 -D CMAKE_INSTALL_PREFIX=./OpenCL-SDK/install -B ./OpenCL-SDK/build -S ./OpenCL-SDK
cmake --build OpenCL-SDK/build --config Release --target install -- /m /v:minimal
```

#### 2.3. Clone and Build CLBlast
- git clone https://github.com/CNugteren/CLBlast
- Navigate to the CLBlast directory:
```bash
cd CLBlast
```

- Create a build directory and navigate into it:
```bash
mkdir build && cd build
```

- Run CMake to configure the project. This command will generate the Visual Studio project files. Make sure to replace C:/GitHub/OpenCL-SDK/install with the correct path where you installed the OpenCL SDK.

```bash
cmake -DOPENCL_INCLUDE_DIRS="C:/GitHub/OpenCL-SDK/install/include" -DOPENCL_LIBRARIES="C:/GitHub/OpenCL-SDK/install/lib/OpenCL.lib" ..
```
- Build the CLBlast project:
```bash
cmake --build . --config Release
```

## 3. Install HDF5 Library
- Download the HDF5 library from the official HDF Group releases page:
  - Visit https://github.com/HDFGroup/hdf5/releases/tag/hdf5_1.14.6
- Download hdf5-1.14.6-win-vs2022_cl.msi
- Install the HDF5 library using the installer.
- This will install the HDF5 library to a default location, typically `C:\Program Files\HDF_Group\HDF5\1.14.6`.
- If you installed it to a different location, make sure to specify the correct path in the CMake configuration step below.

## 4. Configure Project

- In the root directory of OpenCLNeuralNetwork, open CMakeLists.txt and set your OpenCL SDK path:

- For example, I pulled the OpenCL SDK to `C:/GitHub/OpenCL-SDK`, so in the CMakeLists.txt file, the default path is set as follows:
```cmake
set(OPENCL_SDK_PATH "C:/GitHub/OpenCL-SDK/install")
```

- If you installed CLBlast to a custom location, set the path in CMakeLists.txt, I pulled it to `C:/GitHub/CLBlast`, so I set the path as follows:
```cmake
set(CLBLAST_DIR "C:/GitHub/CLBlast")
```

- If you installed HDF5 to a custom location, set the path in CMakeLists.txt:
```cmake
set(HDF5_ROOT "C:/Program Files/HDF_Group/HDF5/1.14.6" CACHE PATH "Path to HDF5 installation directory")
```


## 5. Build the Project
- From the root of the OpenCLNeuralNetwork project:
```bash
rebuild.bat
```

## 6. Run the Program
- After a successful build, you can run the program:
```bash
OpenCLNeuralNetwork.exe
```
- The program will execute the main function defined in `src/main.cpp`, which includes a simple neural network training example.

✅ Done
You're now ready to experiment with the OpenCL-accelerated neural network!




