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
include(rapids-cuda)
include(rapids-cpm)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86)
endif ()
rapids_cuda_init_architectures(PYLIBWHOLEGRAPH)

project(PYLIBWHOLEGRAPH VERSION ${WHOLEGRAPH_VERSION} LANGUAGES C CXX CUDA)

##############################################################################
# - User Options  ------------------------------------------------------------

option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)

##############################################################################
# - Base Rapids Options  -----------------------------------------------------

# default build type
rapids_cmake_build_type(Release)

message("CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

if(DETECT_CONDA_ENV)
    rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
        message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
        set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
    endif()
endif()

rapids_cpm_init()

include(../cmake/thirdparty/get_raft.cmake)

##############################################################################
# - Dependencies  ------------------------------------------------------------

find_package(PythonInterp 3 REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/thirdparty/nanobind.cmake)

# use <package>_ROOT here to take precedence over any other package
set(wholegraph_ROOT "$ENV{LIBWHOLEGRAPH_DIR}")
find_package(wholegraph "${RAPIDS_VERSION}.0" EXACT)
message("WholeGraph")
if (WHOLEGRAPH_FOUND)
    message(STATUS "PYLIBWHOLEGRAPH: using pre-built wholegraph C++ package")
else()
    message(FATAL_ERROR "PYLIBWHOLEGRAPH: could not find wholegraph package in "
            "cmake prefix ${CMAKE_PREFIX_PATH} or user dir $ENV{LIBWHOLEGRAPH_DIR}")
endif()

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import os; import skbuild; print(os.path.join(os.path.dirname(skbuild.__file__), 'resources/cmake'))"
        OUTPUT_VARIABLE SKBUILD_CMAKE_MODULE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_MODULE_PATH "${SKBUILD_CMAKE_MODULE_PATH}")

include(rapids-cython)
rapids_cython_init()

##############################################################################
# - Display options ----------------------------------------------------------

message(VERBOSE "PYLIBWHOLEGRAPH: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")

##############################################################################
# - Compiler options ---------------------------------------------------------

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CXX_DEFINITIONS WHOLEGRAPH_VERSION=${WHOLEGRAPH_VERSION})

message(STATUS "PYLIBWHOLEGRAPH: DEFAULT_CXX_FLAGS='${DEFAULT_CXX_FLAGS}'")
message(STATUS "PYLIBWHOLEGRAPH: CXX_FLAGS='${CXX_FLAGS}'")
message(STATUS "PYLIBWHOLEGRAPH: CXX_DEFINITIONS='${CXX_DEFINITIONS}'")

##############################################################################
# - Variables ----------------------------------------------------------------

#set(WHOLEGRAPH_PY_TARGET "pylibwholegraph_ext" CACHE STRING "wholegraph nanobind target name")

set(WHOLEGRAPH_CPP_TARGET "wholegraph::wholegraph" CACHE STRING "libwholegraph target name")

##############################################################################
# - nanobind targets ---------------------------------------------------------

# Build the actual extension module
#nanobind_add_module(
#        ${WHOLEGRAPH_PY_TARGET}
#        NB_STATIC # Build static libnanobind (the extension module itself remains a shared library)
#        ${CMAKE_CURRENT_LIST_DIR}/cpp/pylibwholegraph_ext.cpp
#        ${CMAKE_CURRENT_LIST_DIR}/cpp/wholegraph_types.cpp
#        ${CMAKE_CURRENT_LIST_DIR}/cpp/wholegraph_functions.cpp
#)
#
## this adds includes from the C++ target as well
#target_link_libraries(${WHOLEGRAPH_PY_TARGET}
#        PUBLIC
#        ${WHOLEGRAPH_CPP_TARGET}
#        )

##############################################################################
# - Install Targets  ---------------------------------------------------------

#install(
#        TARGETS     ${WHOLEGRAPH_PY_TARGET}
#        DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_RELATIVE_SITE_PACKAGES_DIR})

add_subdirectory(pylibwholegraph/binding)

# when used without setup.py, command is like:
# export wholegraph_DIR=`pwd`/../../cpp/build
# cmake ../ -DSKBUILD=ON -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 -DCMAKE_MODULE_PATH=/usr/local/lib/python3.8/dist-packages/skbuild/resources/cmake

add_subdirectory(wholegraph_torch)