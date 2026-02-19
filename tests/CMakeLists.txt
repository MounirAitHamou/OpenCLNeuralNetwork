cmake_minimum_required(VERSION 3.10)
enable_testing()

if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/external/googletest
    ${CMAKE_BINARY_DIR}/googletest_build
)

file(GLOB_RECURSE TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*Test.cpp")

list(LENGTH TEST_FILES TEST_FILES_COUNT)
if(TEST_FILES_COUNT GREATER 0)
    message(STATUS "Found ${TEST_FILES_COUNT} test files.")
else()
    message(WARNING "No test files found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(test_name_list "")
foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)
    string(REPLACE "-" "_" test_name ${test_name})

    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE OpenCLNeuralNetworkLib gtest_main)
    target_include_directories(${test_name} PRIVATE "${CMAKE_SOURCE_DIR}/include")

    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
    list(APPEND test_name_list ${test_name})
endforeach()

if(TEST_FILES_COUNT GREATER 0)
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS ${test_name_list}
        COMMENT "Running all tests..."
    )
endif()
