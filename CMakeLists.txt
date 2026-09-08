cmake_minimum_required(VERSION 3.20)

project(CLNN VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

option(CLNN_BUILD_EXAMPLES "Build CLNN examples" ON)
option(CLNN_BUILD_BENCHMARKS "Build CLNN microbenchmarks" ON)
option(CLNN_BUILD_TESTS "Build CLNN tests" ${BUILD_TESTING})
option(CLNN_BUILD_PYTHON "Build the Python extension module" OFF)
option(CLNN_FETCH_PYBIND11 "Download pybind11 when it is not installed" ON)
option(CLNN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(clnn
    src/autograd/tensor.cpp
    src/autograd/ops.cpp
    src/device.cpp
    src/explain.cpp
    src/data/dataset.cpp
    src/inspect.cpp
    src/nn/module.cpp
    src/opencl/backend.cpp
    src/opencl/opencl_api.cpp
    src/optim/optimizer.cpp
)
add_library(CLNN::clnn ALIAS clnn)
set_target_properties(clnn PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(clnn PUBLIC cxx_std_20)
target_include_directories(clnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(clnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(clnn PRIVATE ${CMAKE_DL_LIBS})

if(MSVC)
    target_compile_options(clnn PRIVATE /W4 /permissive-)
    if(CLNN_WARNINGS_AS_ERRORS)
        target_compile_options(clnn PRIVATE /WX)
    endif()
else()
    target_compile_options(clnn PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(CLNN_WARNINGS_AS_ERRORS)
        target_compile_options(clnn PRIVATE -Werror)
    endif()
endif()

if(CLNN_BUILD_EXAMPLES)
    add_executable(clnn_examples src/main.cpp)
    target_link_libraries(clnn_examples PRIVATE CLNN::clnn)
endif()

if(CLNN_BUILD_BENCHMARKS)
    add_executable(clnn_benchmarks benchmarks/benchmark.cpp)
    target_link_libraries(clnn_benchmarks PRIVATE CLNN::clnn)
endif()

if(CLNN_BUILD_TESTS)
    add_subdirectory(tests)
    if(CLNN_BUILD_BENCHMARKS)
        add_test(NAME clnn_benchmark_regression COMMAND clnn_benchmarks --check)
        set_tests_properties(clnn_benchmark_regression PROPERTIES
            LABELS benchmark
            TIMEOUT 60
        )
    endif()
endif()

if(CLNN_BUILD_PYTHON)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 3.0 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        if(NOT CLNN_FETCH_PYBIND11)
            message(FATAL_ERROR "pybind11 3.x was not found; install it or enable CLNN_FETCH_PYBIND11")
        endif()
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v3.1.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(_clnn MODULE python/bindings.cpp)
    target_link_libraries(_clnn PRIVATE CLNN::clnn)
    target_compile_features(_clnn PRIVATE cxx_std_20)

    set(CLNN_PYTHON_OUTPUT_DIR "${CMAKE_BINARY_DIR}/python/clnn")
    set_target_properties(_clnn PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CLNN_PYTHON_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CLNN_PYTHON_OUTPUT_DIR}"
    )
    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${config}" config_upper)
        set_target_properties(_clnn PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CLNN_PYTHON_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CLNN_PYTHON_OUTPUT_DIR}"
        )
    endforeach()
    add_custom_command(TARGET _clnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CLNN_PYTHON_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/python/clnn"
                "${CLNN_PYTHON_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${CLNN_PYTHON_OUTPUT_DIR}/py.typed"
    )

    install(TARGETS _clnn LIBRARY DESTINATION clnn RUNTIME DESTINATION clnn)
    install(FILES python/clnn/__init__.py python/clnn/__init__.pyi
                  python/clnn/nn.pyi python/clnn/optim.pyi python/clnn/data.pyi
                  python/clnn/explain.pyi
                  python/clnn/py.typed
            DESTINATION clnn)

    if(CLNN_BUILD_TESTS)
        add_test(NAME clnn_python_tests
            COMMAND ${Python_EXECUTABLE} -m unittest discover
                    -s "${CMAKE_CURRENT_SOURCE_DIR}/python/tests" -v
        )
        set_tests_properties(clnn_python_tests PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        )
    endif()
endif()

install(TARGETS clnn EXPORT CLNNTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/clnn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT CLNNTargets FILE CLNNTargets.cmake NAMESPACE CLNN::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CLNN)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CLNNConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    cmake/CLNNConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CLNNConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CLNN
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CLNNConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CLNNConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CLNN
)
