# Build options
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
option(CODE_COVERAGE "Enable generating code coverage with gcov." OFF)

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTestInternal TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_include_directories(${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${WHOLEGRAPH_SOURCE_DIR}>/src")
    target_link_libraries(${TEST_NAME} GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main
            wholememory raft::raft rmm::rmm pthread)
    set_target_properties(
            ${TEST_NAME}
            PROPERTIES POSITION_INDEPENDENT_CODE ON
            RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${WHOLEGRAPH_BINARY_DIR}/gtests>"
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            INSTALL_RPATH "\$ORIGIN/../../../lib")
    target_compile_definitions(${TEST_NAME}
            PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${WHOLEGRAPH_LOGGING_LEVEL}")
    target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Werror
            -Wno-error=deprecated-declarations>)

    if(DISABLE_DEPRECATION_WARNING)
        target_compile_options(
                ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
        target_compile_options(${TEST_NAME}
                PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
    endif()

    if(CODE_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

            set(KEEP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp)
            make_directory(${KEEP_DIR})
            target_compile_options(${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--keep
                    --keep-dir=${KEEP_DIR}>)
            target_compile_options(
                    ${TEST_NAME}
                    PUBLIC
                    $<$<COMPILE_LANGUAGE:CUDA>:-O0
                    -Xcompiler=--coverage,-fprofile-abs-path,-fkeep-inline-functions,-fno-elide-constructors>)
            target_compile_options(
                    ${TEST_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O0 --coverage -fprofile-abs-path
                    -fkeep-inline-functions -fno-elide-constructors>)
            target_link_options(${TEST_NAME} PRIVATE --coverage)
            target_link_libraries(${TEST_NAME} gcov)
        endif()

        # Add coverage-generated files to clean target
        list(APPEND COVERAGE_CLEAN_FILES "**/*.gcno" "**/*.gcda")
        set_property(
                TARGET ${TEST_NAME}
                APPEND
                PROPERTY ADDITIONAL_CLEAN_FILES ${COVERAGE_CLEAN_FILES})
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    install(
            TARGETS ${TEST_NAME}
            COMPONENT testing
            DESTINATION bin/gtests/libwholegraph
            EXCLUDE_FROM_ALL)
endfunction()

# Wrapper around `ConfigureTestInternal` that builds tests both with and without per thread default
# stream
function(ConfigureRMMTest TEST_NAME)
    # Test with legacy default stream.
    ConfigureTestInternal(${TEST_NAME} ${ARGN})

    # Test with per-thread default stream.
    string(REGEX REPLACE "_TEST$" "_PTDS_TEST" PTDS_TEST_NAME "${TEST_NAME}")
    ConfigureTestInternal("${PTDS_TEST_NAME}" ${ARGN})
    target_compile_definitions("${PTDS_TEST_NAME}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

    # Test with custom thrust namespace
    string(REGEX REPLACE "_TEST$" "_NAMESPACE_TEST" NS_TEST_NAME "${TEST_NAME}")
    ConfigureTestInternal("${NS_TEST_NAME}" ${ARGN})
    target_compile_definitions("${NS_TEST_NAME}" PUBLIC THRUST_WRAPPED_NAMESPACE=wholegraph_thrust)
endfunction()

# Wrapper around `ConfigureTestInternal` that builds tests both with and without per thread default
# stream
function(ConfigureTest TEST_NAME)
    # Test with legacy default stream.
    ConfigureTestInternal(${TEST_NAME} ${ARGN})
endfunction()

# pinned buffer tests
ConfigureRMMTest(PINNED_BUFFER_TEST rmm/pinned_buffer_tests.cu)
#ConfigureTest(PINNED_BUFFER_TEST rmm/pinned_buffer_tests.cu)

# uvector tests
ConfigureRMMTest(PINNED_UVECTOR_TEST rmm/pinned_uvector_tests.cpp)
#ConfigureTest(PINNED_UVECTOR_TEST rmm/pinned_uvector_tests.cpp)

# parallel_utils tests
ConfigureTest(PARALLEL_UTILS_TEST parallel_utils_tests.cpp)

# wholememory communicator tests
ConfigureTest(WHOLEMEMORY_COMM_TEST wholememory/wholememory_comm_tests.cpp)

# wholememory handle tests
ConfigureTest(WHOLEMEMORY_HANDLE_TEST wholememory/wholememory_handle_tests.cpp)

# wholememory gather op tests
ConfigureTest(WHOLEMEMORY_GATHER_TEST wholememory_ops/wholememory_gather_tests.cu wholememory_ops/embedding_test_utils.cu)