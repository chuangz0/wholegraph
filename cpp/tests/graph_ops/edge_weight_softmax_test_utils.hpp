/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <wholememory/tensor_description.h>

namespace graph_ops {
namespace testing {
void host_edge_weight_softmax_forward(void* host_csr_row_ptr,
                                      wholememory_array_description_t csr_row_ptr_array_desc,
                                      void* host_weight_ptr,
                                      wholememory_matrix_description_t weight_matrix_desc,
                                      void* host_ref_output_weight_ptr,
                                      wholememory_matrix_description_t output_weight_matrix_desc);
void host_edge_weight_softmax_backward(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* host_weight_ptr,
  wholememory_matrix_description_t weight_matrix_desc,
  void* host_grad_weight_softmax_ptr,
  wholememory_matrix_description_t grad_weight_softmax_matrix_desc,
  void* host_ref_output_grad_weight_ptr,
  wholememory_matrix_description_t output_grad_weight_matrix_desc);
}  // namespace testing
}  // namespace graph_ops
