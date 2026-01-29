cmake_minimum_required(VERSION 3.10)

project(OpenCLNeuralNetworkLib LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
cmake_policy(SET CMP0074 NEW)

if(NOT DEFINED OpenCL_ROOT)
    set(OpenCL_ROOT "C:/OpenCL-SDK" CACHE PATH "Path to OpenCL SDK root")
endif()

set(OpenCL_INCLUDE_DIRS "${OpenCL_ROOT}/include")

find_library(OpenCL_LIBRARY
    NAMES OpenCL
    PATHS "${OpenCL_ROOT}/lib" "${OpenCL_ROOT}/lib/x64"
    NO_DEFAULT_PATH
)

if(NOT OpenCL_LIBRARY)
    message(FATAL_ERROR "OpenCL library not found in ${OpenCL_ROOT}/lib")
endif()

if(NOT DEFINED HDF5_ROOT)
    set(HDF5_ROOT "C:/Program Files/HDF_Group/HDF5/1.14.6" CACHE PATH "Path to HDF5")
endif()
find_package(HDF5 COMPONENTS CXX REQUIRED)
if(NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found!")
endif()

set(CLBLAST_DIR "C:/CLBlast" CACHE PATH "Path to CLBlast root")
set(CLBLAST_INCLUDE_DIRS "${CLBLAST_DIR}/include")
set(CLBLAST_LIBRARY "${CLBLAST_DIR}/lib/clblast.lib")

add_library(OpenCLNeuralNetworkLib STATIC
    src/DataLoaders/CSVNumerical/CSVNumericalLoader.cpp
    src/DataLoaders/BinImage/BinImageDataLoader.cpp
    src/DataLoaders/DataLoader.cpp

    src/Layers/ActivationLayers/PreActivationLayers/LeakyReLU/LeakyReLULayer.cpp
    src/Layers/ActivationLayers/PreActivationLayers/ReLU/ReLULayer.cpp
    src/Layers/ActivationLayers/Sigmoid/SigmoidLayer.cpp
    src/Layers/ActivationLayers/Tanh/TanhLayer.cpp
    src/Layers/ActivationLayers/Softmax/SoftmaxLayer.cpp

    src/Layers/TrainableLayers/Convolutional/ConvolutionalLayer.cpp
    src/Layers/TrainableLayers/Dense/DenseLayer.cpp

    src/NeuralNetworks/Local/LocalNeuralNetwork.cpp

    src/Optimizers/AdamBaseOptimizers/Adam/AdamOptimizer.cpp
    src/Optimizers/AdamBaseOptimizers/AdamW/AdamWOptimizer.cpp
    src/Optimizers/SGD/SGDOptimizer.cpp

    src/LossFunctions/BinaryCrossEntropy/BinaryCrossEntropy.cpp
    src/LossFunctions/MeanSquaredError/MeanSquaredError.cpp
    src/LossFunctions/CategoricalCrossEntropy/CategoricalCrossEntropy.cpp
    src/LossFunctions/SoftmaxCrossEntropy/SoftmaxCrossEntropy.cpp

    src/Utils/EventProfiler.cpp
    src/Utils/LayerArgs.cpp
    src/Utils/NetworkArgs.cpp
    src/Utils/OpenCLResources.cpp
    src/Utils/OptimizerArgs.cpp
    src/Utils/LossFunctionArgs.cpp
)

target_include_directories(OpenCLNeuralNetworkLib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${CLBLAST_INCLUDE_DIRS}
)

target_compile_definitions(OpenCLNeuralNetworkLib PUBLIC
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    ${HDF5_DEFINITIONS}
)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD /W3")
endif()

target_link_libraries(OpenCLNeuralNetworkLib PRIVATE
    ${OpenCL_LIBRARY}
    ${HDF5_CXX_LIBRARIES}
    ${CLBLAST_LIBRARY}
)

add_executable(OpenCLNeuralNetwork
    src/main.cpp
)

set_target_properties(OpenCLNeuralNetwork PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_link_libraries(OpenCLNeuralNetwork PRIVATE
    OpenCLNeuralNetworkLib
)

target_compile_definitions(OpenCLNeuralNetworkLib PUBLIC
    KERNELS_DIR="${CMAKE_SOURCE_DIR}/kernels"
)

add_custom_command(
    TARGET OpenCLNeuralNetwork POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLBLAST_DIR}/bin/clblast.dll"
        "${CMAKE_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HDF5_ROOT}/bin/hdf5_cpp.dll"
        "${CMAKE_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HDF5_ROOT}/bin/hdf5.dll"
        "${CMAKE_SOURCE_DIR}"
    COMMENT "Copying required DLLs (CLBlast + HDF5) to root folder"
)

add_subdirectory(tests)
