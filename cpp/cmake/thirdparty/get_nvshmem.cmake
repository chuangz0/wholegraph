#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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


set(USE_NVSHMEM_VERSION 2.10.1)
set(USE_NVSHMEM_VERSION_BRANCH 3)
set(NVSHMEM_BOOTSTRAP_PLUGIN_DIR "" )
set(CPM_NVSHMEM_USED OFF)
function(find_and_configure_nvshmem)


    set(oneValueArgs VERSION VERSION_BRANCH   EXCLUDE_FROM_ALL INSTALL_DIR DOWNLOAD)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )


    if(NOT ${PKG_DOWNLOAD})

        rapids_find_generate_module(nvshmem
            HEADER_NAMES  nvshmem.h
            PATHS  ${NVSHMEM_HOME}/include
            LIBRARY_NAMES nvshmem_host nvshmem_device
            PATHS ${NVSHMEM_HOME}/lib PATH_SUFFIXES .a .so
        )
        rapids_find_package(nvshmem ${PKG_VERSION})
        set(NVSHMEM_BOOTSTRAP_PLUGIN_DIR ${nvshmem_INCLUDE_DIR}/../share/nvshmem/src/bootstrap-plugins PARENT_SCOPE)

    endif()

    if( (NOT DEFINED NVSHMEM_FOUND) OR (NOT  ${NVSHMEM_FOUND}))
        rapids_cpm_find(nvshmem ${PKG_VERSION}
                    GLOBAL_TARGETS nvshmem::nvshmem nvshmem::nvshmem_device nvshmem::nvshmem_host
                    CPM_ARGS
                        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
                        URL https://developer.download.nvidia.cn/compute/redist/nvshmem/${PKG_VERSION}/source/nvshmem_src_${PKG_VERSION}-${PKG_VERSION_BRANCH}.txz
                        OPTIONS
                            "NVSHMEM_IBGDA_SUPPORT ON"
                            "NVSHMEM_IBDEVX_SUPPORT ON"
                            "NVSHMEM_BUILD_EXAMPLES OFF"
                            "NVSHMEM_BUILD_TESTS OFF"
                            "NVSHMEM_PREFIX ${PKG_INSTALL_DIR}"
                    )
        if(NOT TARGET nvshmem::nvshmem AND TARGET nvshmem)
            add_library( nvshmem::nvshmem ALIAS nvshmem)
            add_library(nvshmem::nvshmem_device ALIAS nvshmem_device)
            add_library(nvshmem::nvshmem_host ALIAS nvshmem_host)
        endif()
        set(NVSHMEM_BOOTSTRAP_PLUGIN_DIR ${nvshmem_BINARY_DIR}/src/include/modules/bootstrap  ${nvshmem_BINARY_DIR}/src/include/modules/common PARENT_SCOPE)
        set(CPM_NVSHMEM_USED ON PARENT_SCOPE)
    endif()


endfunction()


find_and_configure_nvshmem(VERSION           ${USE_NVSHMEM_VERSION}
                           VERSION_BRANCH    ${USE_NVSHMEM_VERSION_BRANCH}
                           EXCLUDE_FROM_ALL  ${WHOLEGRAPH_EXCLUDE_NVSHMEM_FROM_ALL}
                           INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
                           DOWNLOAD          ${DOWNLOAD_NVSHMEM}
                           )
