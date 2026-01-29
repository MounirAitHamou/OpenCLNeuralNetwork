cmake_minimum_required(VERSION 3.10)

enable_testing()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/external/googletest
    ${CMAKE_BINARY_DIR}/googletest_build
)

file(GLOB_RECURSE TEST_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*Test.cpp"
)

list(LENGTH TEST_FILES TEST_FILES_COUNT)

if(TEST_FILES_COUNT GREATER 0)
    message(STATUS "Found ${TEST_FILES_COUNT} test files.")
else()
    message(WARNING "No test files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(TEST_TARGETS "")

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)

    string(REPLACE "-" "_" test_name ${test_name})

    add_executable(${test_name} ${test_file})

    target_link_libraries(${test_name} PRIVATE
        OpenCLNeuralNetworkLib
        gtest_main
    )

    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    add_test(NAME ${test_name} COMMAND ${test_name})

    list(APPEND TEST_TARGETS ${test_name})
endforeach()

if(TEST_TARGETS)
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${TEST_TARGETS}
        COMMENT "Running all tests..."
    )
endif()

set(CLBLAST_DLL "${CLBLAST_DIR}/bin/clblast.dll")
set(TEST_BIN_DIR "${CMAKE_BINARY_DIR}/bin")

add_custom_target(copy_runtime_dlls ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLBLAST_DLL}"
        "${TEST_BIN_DIR}"
    COMMENT "Copying CLBlast runtime DLL to test bin directory"
)