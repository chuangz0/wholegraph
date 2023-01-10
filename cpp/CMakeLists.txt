#=============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

set(RAPIDS_VERSION "22.12")
set(WHOLEGRAPH_VERSION "${RAPIDS_VERSION}.00")

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  rapids-cmake
  GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
  GIT_TAG        origin/branch-${RAPIDS_VERSION}
)
FetchContent_MakeAvailable(rapids-cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86)
endif ()
rapids_cuda_init_architectures(WHOLEGRAPH)

project(WHOLEGRAPH VERSION ${WHOLEGRAPH_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_USE_RESPONSE_FILE_FOR_INCLUDES   FALSE)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES   FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES   FALSE)

# Write the version header
rapids_cmake_write_version_file(include/wholegraph/version_config.hpp)

##############################################################################
# - User Options  ------------------------------------------------------------
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)

##############################################################################
# - Set options based on user defined one  -----------------------------------

##############################################################################
# - Base rapids options ------------------------------------------------------
# default build type
#rapids_cmake_build_type(Release)
rapids_cmake_build_type(RelWithDebInfo)

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

################################################################################
# - compiler options -----------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET    wholegraph-exports
  INSTALL_EXPORT_SET  wholegraph-exports
)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WHOLEGRAPH_CXX_FLAGS "")
set(WHOLEGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND WHOLEGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

#list(APPEND WHOLEGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
#list(APPEND WHOLEGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
#list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
#list(APPEND WHOLEGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND WHOLEGRAPH_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND WHOLEGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()

################################################################################
# - find CPM based dependencies  -----------------------------------------------

rapids_cpm_init()

include(../cmake/thirdparty/get_raft.cmake)
include(../cmake/thirdparty/get_nccl.cmake)

file(GLOB WHOLEMEMORY_SOURCES
        "src/*.cpp"
        "src/wholememory/*.cpp"
        "src/wholememory_ops/*.cpp"
        "src/wholememory_ops/*.cu"
        "src/wholememory_ops/functions/*.cu")

add_library(wholememory SHARED "")

target_sources(wholememory PRIVATE ${WHOLEMEMORY_SOURCES})

set_target_properties(wholememory
        PROPERTIES BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
        # set target compile options
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
        INTERFACE_POSITION_INDEPENDENT_CODE ON
        )

target_compile_options(wholememory
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${WHOLEGRAPH_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${WHOLEGRAPH_CUDA_FLAGS}>"
)

target_link_libraries(wholememory PRIVATE -static-libgcc -static-libstdc++)

################################################################################
# - include paths --------------------------------------------------------------

target_include_directories(wholememory
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)


################################################################################
# - link libraries -------------------------------------------------------------
target_link_libraries(wholememory
PUBLIC
    raft::raft
    CUDA::cuda_driver
    CUDA::cudart
PRIVATE
    NCCL::NCCL
)

# optionally build tests
if(BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(../cmake/thirdparty/get_gtest.cmake)
    include(CTest) # calls enable_testing()

    add_subdirectory(tests)
endif()

