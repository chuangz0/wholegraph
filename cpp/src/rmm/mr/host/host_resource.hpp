/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/mr/host/host_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

namespace rmm::mr {

/**
 * @brief Returns a pointer to the pinned memory resource.
 *
 * Returns a global instance of a `pinned_memory_resource` as a function local static.
 *
 * @return Pointer to the static pinned_memory_resource used as the initial, default resource
 */
inline pinned_memory_resource* get_pinned_resource()
{
  static pinned_memory_resource mr{};
  return &mr;
}

}  // namespace rmm::mr
