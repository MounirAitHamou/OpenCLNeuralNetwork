cmake_minimum_required(VERSION 3.10)

project(OpenCLNeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

set(SDKINSTALLROOT "C:/GitHub/OpenCL-SDK/install")

add_executable(OpenCLNeuralNetwork
    src/main.cpp
    src/Layer/Dense/DenseLayer.cpp
    src/NeuralNetwork/NeuralNetwork.cpp
    src/Utils/Dimensions.cpp
    src/Utils/OpenCLSetup.cpp
    src/Optimizer/SGD/SGDOptimizer.cpp
    src/Optimizer/Adam/AdamOptimizer.cpp
    src/Optimizer/AdamW/AdamWOptimizer.cpp
    src/Utils/LayerConfig.cpp
    src/Utils/OptimizerConfig.cpp
)

target_include_directories(OpenCLNeuralNetwork PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${SDKINSTALLROOT}/include"
)

target_compile_definitions(OpenCLNeuralNetwork PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
)

if(MSVC)
    target_compile_options(OpenCLNeuralNetwork PRIVATE /W3)
endif()

target_link_directories(OpenCLNeuralNetwork PRIVATE "${SDKINSTALLROOT}/lib")

target_link_libraries(OpenCLNeuralNetwork PRIVATE OpenCL.lib)