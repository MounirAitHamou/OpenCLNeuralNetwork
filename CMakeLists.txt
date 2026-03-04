cmake_minimum_required(VERSION 3.10)
project(OpenCLNeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
cmake_policy(SET CMP0074 NEW)

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

if(MSVC)
    foreach(flag_var
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE)
        string(REPLACE "/W3" "" ${flag_var} "${${flag_var}}")
    endforeach()
endif()

find_package(OpenCL REQUIRED)
find_package(CLBlast CONFIG REQUIRED)
find_package(HDF5 CONFIG REQUIRED COMPONENTS C CXX)

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
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_definitions(OpenCLNeuralNetworkLib PUBLIC
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    KERNELS_DIR="${PROJECT_SOURCE_DIR}/kernels"
)

set(HDF5_LINK_LIBRARIES "")

if(TARGET hdf5::hdf5_cpp-shared)
    list(APPEND HDF5_LINK_LIBRARIES hdf5::hdf5-shared hdf5::hdf5_cpp-shared)
elseif(TARGET hdf5::hdf5_cpp-static)
    list(APPEND HDF5_LINK_LIBRARIES hdf5::hdf5-static hdf5::hdf5_cpp-static)
elseif(TARGET hdf5::hdf5_cpp)
    list(APPEND HDF5_LINK_LIBRARIES hdf5::hdf5 hdf5::hdf5_cpp)
else()
    list(APPEND HDF5_LINK_LIBRARIES ${HDF5_LIBRARIES})
endif()

target_link_libraries(OpenCLNeuralNetworkLib PUBLIC
    OpenCL::OpenCL
    clblast
    ${HDF5_LINK_LIBRARIES}
)

target_include_directories(OpenCLNeuralNetworkLib PUBLIC
    ${HDF5_INCLUDE_DIRS}
)

add_executable(OpenCLNeuralNetwork src/main.cpp)

target_compile_definitions(OpenCLNeuralNetwork PRIVATE
    DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

target_link_libraries(OpenCLNeuralNetwork PRIVATE
    OpenCLNeuralNetworkLib
)

if (MSVC)

    target_compile_options(OpenCLNeuralNetworkLib PRIVATE
        /W4
        /permissive-
        /EHsc
        $<$<CONFIG:Debug>:/fsanitize=address>
        $<$<CONFIG:Debug>:/RTC1>
        $<$<CONFIG:Debug>:/Zi>
    )

    target_link_options(OpenCLNeuralNetworkLib PRIVATE
        $<$<CONFIG:Debug>:/fsanitize=address>
    )

    target_compile_options(OpenCLNeuralNetwork PRIVATE
        /W4
        /permissive-
        /EHsc
        $<$<CONFIG:Debug>:/fsanitize=address>
        $<$<CONFIG:Debug>:/RTC1>
        $<$<CONFIG:Debug>:/Zi>
    )

    target_link_options(OpenCLNeuralNetwork PRIVATE
        $<$<CONFIG:Debug>:/fsanitize=address>
    )

else()

    target_compile_options(OpenCLNeuralNetworkLib PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-g>
    )

    target_link_options(OpenCLNeuralNetworkLib PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )

    target_compile_options(OpenCLNeuralNetwork PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-g>
    )

    target_link_options(OpenCLNeuralNetwork PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )

endif()

if(WIN32 AND MSVC)
    set(VCPKG_BIN_DIR "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin")
    set_target_properties(OpenCLNeuralNetwork PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "PATH=%VCPKG_BIN_DIR%;$ENV{PATH}"
    )
endif()

add_subdirectory(tests)