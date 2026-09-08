if(EXISTS "${PROJECT_SOURCE_DIR}/external/googletest/CMakeLists.txt")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${PROJECT_SOURCE_DIR}/external/googletest"
        "${PROJECT_BINARY_DIR}/googletest"
        EXCLUDE_FROM_ALL
    )
else()
    find_package(GTest CONFIG REQUIRED)
endif()

add_executable(clnn_tests
    autograd_test.cpp
    data_checkpoint_test.cpp
    losses_test.cpp
    module_optimizer_test.cpp
    nn_features_test.cpp
    opencl_test.cpp
    tensor_ops_edge_test.cpp
    data_edge_test.cpp
    explain_test.cpp
    serialization_optimizer_edge_test.cpp
)
target_link_libraries(clnn_tests PRIVATE CLNN::clnn GTest::gtest_main)
target_compile_features(clnn_tests PRIVATE cxx_std_20)
target_compile_definitions(clnn_tests PRIVATE CLNN_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(clnn_tests)
