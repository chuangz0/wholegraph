#pragma once

#include <torch/script.h>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

namespace wholegraph_torch {

c10::ScalarType get_c10_scalar_type(wholememory_dtype_t wm_dtype);

wholememory_dtype_t get_wholememory_dtype(caffe2::TypeMeta th_dtype);

struct pytorch_memory_context {
  torch::Tensor tensor;
  torch::TensorOptions options;
};

void set_need_grad(memory_context_t* memory_context, bool require_grad);

void create_torch_memory_context_func(memory_context_t* memory_context, void* /*global_context*/);

void destroy_torch_memory_context_func(memory_context_t* memory_context, void* /*global_context*/);

void *torch_common_malloc_func(wholememory_tensor_description_t *tensor_description,
                               memory_context_t *memory_context,
                               bool gpu_memory = true,
                               bool pinned = false);

void torch_common_free_func(memory_context_t* memory_context, void* /*global_context*/);

void get_tensor_desc_from_torch_tensor(wholememory_tensor_description_t* tensor_desc, const torch::Tensor& t);

void get_array_desc_from_torch_tensor(wholememory_array_description_t* array_desc, const torch::Tensor& t);

void get_matrix_desc_from_torch_tensor(wholememory_matrix_description_t* matrix_desc, const torch::Tensor& t);

}  // namespace wholegraph_torch