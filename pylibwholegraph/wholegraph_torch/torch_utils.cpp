#include "torch_utils.h"

#include <c10/cuda/CUDAFunctions.h>

namespace wholegraph_torch {

c10::ScalarType get_c10_scalar_type(wholememory_dtype_t wm_dtype) {
  switch(wm_dtype) {
    case WHOLEMEMORY_DT_FLOAT: return c10::ScalarType::Float;
    case WHOLEMEMORY_DT_HALF: return c10::ScalarType::Half;
    case WHOLEMEMORY_DT_DOUBLE: return c10::ScalarType::Double;
    case WHOLEMEMORY_DT_BF16: return c10::ScalarType::BFloat16;
    case WHOLEMEMORY_DT_INT: return c10::ScalarType::Int;
    case WHOLEMEMORY_DT_INT64: return c10::ScalarType::Long;
    case WHOLEMEMORY_DT_INT16: return c10::ScalarType::Short;
    case WHOLEMEMORY_DT_INT8: return c10::ScalarType::Char;
    default: return c10::ScalarType::Undefined;
  }
}

wholememory_dtype_t get_wholememory_dtype(caffe2::TypeMeta th_dtype) {
  switch (th_dtype.toScalarType()) {
    case c10::ScalarType::Float: return WHOLEMEMORY_DT_FLOAT;
    case c10::ScalarType::Half: return WHOLEMEMORY_DT_HALF;
    case c10::ScalarType::Double: return WHOLEMEMORY_DT_DOUBLE;
    case c10::ScalarType::BFloat16: return WHOLEMEMORY_DT_BF16;
    case c10::ScalarType::Int: return WHOLEMEMORY_DT_INT;
    case c10::ScalarType::Long: return WHOLEMEMORY_DT_INT64;
    case c10::ScalarType::Short: return WHOLEMEMORY_DT_INT16;
    case c10::ScalarType::Char: return WHOLEMEMORY_DT_INT8;
    default: return WHOLEMEMORY_DT_UNKNOWN;
  }
}

void set_need_grad(memory_context_t* memory_context, bool require_grad) {
  static_cast<pytorch_memory_context *>(memory_context->context)->options =
      static_cast<pytorch_memory_context *>(memory_context->context)->options.requires_grad(require_grad);
}

void create_torch_memory_context_func(memory_context_t* memory_context, void* /*global_context*/) {
  wholememory_initialize_tensor_desc(&memory_context->desc);
  memory_context->context = new pytorch_memory_context();
}

void destroy_torch_memory_context_func(memory_context_t* memory_context, void* /*global_context*/) {
  wholememory_initialize_tensor_desc(&memory_context->desc);
  if (memory_context->context != nullptr) {
    delete static_cast<pytorch_memory_context *>(memory_context->context);
    memory_context->context = nullptr;
  }
}

void *torch_common_malloc_func(wholememory_tensor_description_t *tensor_description,
                               memory_context_t *memory_context,
                               bool gpu_memory,
                               bool pinned) {
  memory_context->desc = *tensor_description;
  auto* pytorch_context = static_cast<pytorch_memory_context *>(memory_context->context);
  std::vector<int64_t> shape(tensor_description->dim);
  for (int i = 0; i < tensor_description->dim; i++) {
    shape[i] = tensor_description->sizes[i];
  }
  pytorch_context->options = pytorch_context->options.dtype(get_c10_scalar_type(tensor_description->dtype));
  if (gpu_memory) {
    pytorch_context->options = pytorch_context->options.device(c10::Device(c10::kCUDA, c10::cuda::current_device()));
  } else {
    pytorch_context->options = pytorch_context->options.device(c10::Device(c10::kCPU));
    pytorch_context->options = pytorch_context->options.pinned_memory(pinned);
  }
  pytorch_context->tensor = torch::empty(shape, pytorch_context->options);
  return pytorch_context->tensor.data_ptr();
}

void torch_common_free_func(memory_context_t* memory_context, void* /*global_context*/)
{
  wholememory_initialize_tensor_desc(&memory_context->desc);
  static_cast<pytorch_memory_context *>(memory_context->context)->tensor = torch::Tensor();
  static_cast<pytorch_memory_context *>(memory_context->context)->options = torch::TensorOptions();
}

void get_tensor_desc_from_torch_tensor(wholememory_tensor_description_t* tensor_desc, const torch::Tensor& t) {
  tensor_desc->dim = t.dim();
  tensor_desc->dtype = get_wholememory_dtype(t.dtype());
  TORCH_CHECK(tensor_desc->dtype != WHOLEMEMORY_DT_UNKNOWN);
  tensor_desc->storage_offset = t.storage_offset();
  for (int i = 0; i < tensor_desc->dim; i++) {
    tensor_desc->sizes[i] = t.size(i);
    tensor_desc->strides[i] = t.stride(i);
  }
}

void get_array_desc_from_torch_tensor(wholememory_array_description_t* array_desc, const torch::Tensor& t) {
  TORCH_CHECK(t.dim() == 1, "get_array_desc_from_torch_tensor: should be 1-dim tensor");
  array_desc->dtype = get_wholememory_dtype(t.dtype());
  TORCH_CHECK(array_desc->dtype != WHOLEMEMORY_DT_UNKNOWN);
  array_desc->size = t.size(0);
  array_desc->storage_offset = t.storage_offset();
}

void get_matrix_desc_from_torch_tensor(wholememory_matrix_description_t* matrix_desc, const torch::Tensor& t) {
  TORCH_CHECK(t.dim() == 2, "get_matrix_desc_from_torch_tensor: should be 2-dim tensor");
  matrix_desc->dtype = get_wholememory_dtype(t.dtype());
  TORCH_CHECK(matrix_desc->dtype != WHOLEMEMORY_DT_UNKNOWN);
  matrix_desc->sizes[0] = t.size(0);
  matrix_desc->sizes[1] = t.size(1);
  matrix_desc->stride = t.stride(1);
  matrix_desc->storage_offset = t.storage_offset();
}

}  // namespace wholegraph_torch