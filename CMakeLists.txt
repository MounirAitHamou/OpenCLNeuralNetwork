cmake_minimum_required(VERSION 3.10)

project(OpenCLNeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

cmake_policy(SET CMP0074 NEW)

set(HDF5_ROOT "C:/Program Files/HDF_Group/HDF5/1.14.6" CACHE PATH "Path to HDF5 installation directory")

find_package(HDF5 COMPONENTS CXX REQUIRED)

if (NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found! Please ensure it's installed and findable by CMake.")
endif()

set(SDKINSTALLROOT "C:/GitHub/OpenCL-SDK/install")
set(CLBLAST_DIR "C:/GitHub/CLBlast")
set(CLBLAST_INCLUDE_DIRS "${CLBLAST_DIR}/include")
set(CLBLAST_LIBRARY_PATH "${CLBLAST_DIR}/build/Release/clblast.lib")

get_filename_component(CLBLAST_LIBRARY_DIR "${CLBLAST_LIBRARY_PATH}" DIRECTORY)

add_executable(OpenCLNeuralNetwork
    src/main.cpp
    src/Layer/Dense/DenseLayer.cpp
    src/Layer/Convolutional/ConvolutionalLayer.cpp
    src/NeuralNetwork/NeuralNetwork.cpp
    src/Utils/OpenCLResources.cpp
    src/Optimizer/SGD/SGDOptimizer.cpp
    src/Optimizer/Adam/AdamOptimizer.cpp
    src/Optimizer/AdamW/AdamWOptimizer.cpp
    src/Utils/LayerArgs.cpp
    src/Utils/OptimizerArgs.cpp
    src/Utils/NetworkArgs.cpp
    src/DataLoader/DataLoader.cpp
    src/DataLoader/CSVNumerical/CSVNumericalLoader.cpp
    src/Utils/EventProfiler.cpp
)

target_include_directories(OpenCLNeuralNetwork PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${SDKINSTALLROOT}/include"
    ${HDF5_INCLUDE_DIRS}
    "${CLBLAST_INCLUDE_DIRS}"
)

target_compile_definitions(OpenCLNeuralNetwork PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    ${HDF5_DEFINITIONS}
)

if(MSVC)
    target_compile_options(OpenCLNeuralNetwork PRIVATE /W3)
endif()

target_link_directories(OpenCLNeuralNetwork PRIVATE
    "${SDKINSTALLROOT}/lib"
    "${CLBLAST_LIBRARY_DIR}"
)

target_link_libraries(OpenCLNeuralNetwork PRIVATE
    OpenCL.lib
    ${HDF5_CXX_LIBRARIES}
    clblast.lib
)

add_custom_command(
    TARGET OpenCLNeuralNetwork POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CLBLAST_DIR}/build/Release/clblast.dll"
    $<TARGET_FILE_DIR:OpenCLNeuralNetwork>
    COMMENT "Copying clblast.dll to the executable directory"
)