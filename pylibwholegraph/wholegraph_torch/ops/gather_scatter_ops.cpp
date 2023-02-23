#include <torch/script.h>

#include <c10/cuda/CUDAStream.h>
#include <wholememory/wholememory_op.h>
#include <wholememory/wholememory_tensor.h>

#include "../torch_env_func_ptrs.h"
#include "../torch_utils.h"

namespace wholegraph_torch {

torch::Tensor gather(int64_t wholememory_tensor_handle,
                     const torch::Tensor &indices,
                     torch::optional<torch::ScalarType> output_type,
                     torch::optional<bool> requires_grad) {
  TORCH_CHECK(indices.dim() == 1, "indices should be 1-dim tensor.")
  TORCH_CHECK(indices.dtype() == torch::kInt32 || indices.dtype() == torch::kInt64,
              "indices should be IntTensor or LongTensor.")
  auto wt = reinterpret_cast<wholememory_tensor_t>(wholememory_tensor_handle);
  wholememory_tensor_description_t wm_tensor_desc;
  wholememory_tensor_get_tensor_description(&wm_tensor_desc, wt);
  TORCH_CHECK(wm_tensor_desc.dim == 1 || wm_tensor_desc.dim == 2,
              "wholememory_tensor_handle should be 1D or 2D WholeMemory Tensor.")
  wholememory_matrix_description_t wm_desc;
  if (wm_tensor_desc.dim == 1) {
    wholememory_array_description_t wm_array_desc;
    TORCH_CHECK(wholememory_convert_tensor_desc_to_array(&wm_array_desc, &wm_tensor_desc),
                "wholememory_tensor_handle convert to array failed.")
    wholememory_copy_array_desc_to_matrix(&wm_desc, &wm_array_desc);
  } else {
    TORCH_CHECK(wholememory_convert_tensor_desc_to_matrix(&wm_desc, &wm_tensor_desc),
                "wholememory_tensor_handle convert to matrix failed.")
  }
  wholememory_array_description_t indices_desc;
  get_array_desc_from_torch_tensor(&indices_desc, indices);

  wholememory_dtype_t wm_output_type = wm_desc.dtype;
  if (output_type.has_value()) {
    wm_output_type = get_wholememory_dtype(output_type.value());
  }

  int64_t output_size[2] = {indices_desc.size, wm_desc.sizes[1]};
  wholememory_matrix_description_t output_desc = wholememory_create_matrix_desc(output_size,
                                                                                wm_desc.sizes[1],
                                                                                0,
                                                                                wm_output_type);
  memory_context_t output_context;
  create_torch_memory_context_func(&output_context, nullptr);
  if (requires_grad.has_value()) {
    set_need_grad(&output_context, requires_grad.value());
  }
  wholememory_tensor_description_t output_alloc_tensor_desc;
  wholememory_copy_matrix_desc_to_tensor(&output_alloc_tensor_desc, &output_desc);
  if (wm_tensor_desc.dim == 1) {
    output_alloc_tensor_desc.dim = 1;
  }
  torch_common_malloc_func(&output_alloc_tensor_desc, &output_context);

  auto output_tensor = static_cast<pytorch_memory_context*>(output_context.context)->tensor;
  destroy_torch_memory_context_func(&output_context, nullptr);

  TORCH_CHECK(wholememory_gather(wt,
                                 indices.data_ptr(),
                                 indices_desc,
                                 output_tensor.data_ptr(),
                                 output_desc,
                                 wholegraph_torch::get_pytorch_env_func(),
                                 wholegraph_torch::get_current_stream()) == WHOLEMEMORY_SUCCESS)
  return output_tensor;
}

void scatter(const torch::Tensor &input,
             const torch::Tensor &indices,
             int64_t wholememory_tensor_handle) {
  TORCH_CHECK(input.dim() == 1 || input.dim() == 2, "input should be 1D or 2D tensor")
  TORCH_CHECK(indices.dim() == 1, "indices should be 1-D tensor.")
  TORCH_CHECK(indices.dtype() == torch::kInt32 || indices.dtype() == torch::kInt64,
              "indices should be IntTensor or LongTensor.")
  auto wt = reinterpret_cast<wholememory_tensor_t>(wholememory_tensor_handle);
  wholememory_tensor_description_t wm_tensor_desc;
  wholememory_tensor_get_tensor_description(&wm_tensor_desc, wt);
  TORCH_CHECK(wm_tensor_desc.dim == input.dim(), "input and wholememory_tensor_hand should be same dim.")
  wholememory_matrix_description_t wm_desc;
  if (wm_tensor_desc.dim == 1) {
    wholememory_array_description_t wm_array_desc;
    TORCH_CHECK(wholememory_convert_tensor_desc_to_array(&wm_array_desc, &wm_tensor_desc),
                "wholememory_tensor_handle convert to array failed.")
    wholememory_copy_array_desc_to_matrix(&wm_desc, &wm_array_desc);
  } else {
    TORCH_CHECK(wholememory_convert_tensor_desc_to_matrix(&wm_desc, &wm_tensor_desc),
                "wholememory_tensor_handle convert to matrix failed.")
  }
  wholememory_array_description_t indices_desc;
  get_array_desc_from_torch_tensor(&indices_desc, indices);
  wholememory_matrix_description_t input_desc;
  if (input.dim() == 1) {
    wholememory_array_description_t input_array_desc;
    get_array_desc_from_torch_tensor(&input_array_desc, input);
    wholememory_copy_array_desc_to_matrix(&input_desc, &input_array_desc);
  } else {
    get_matrix_desc_from_torch_tensor(&input_desc, input);
  }
  TORCH_CHECK(input.size(1) == wm_desc.sizes[1],
              "input and wholememory should have same embedding size but input.size(1)=%ld, wholememory.size(1)=%ld",
              input.size(1), wm_desc.sizes[1])

  TORCH_CHECK(wholememory_scatter(input.data_ptr(),
                                  input_desc,
                                  indices.data_ptr(),
                                  indices_desc,
                                  wt,
                                  wholegraph_torch::get_pytorch_env_func(),
                                  wholegraph_torch::get_current_stream()) == WHOLEMEMORY_SUCCESS)
}

}  // namespace wholegraph_torch

static auto registry =
    torch::RegisterOperators()
      .op("wholegraph::gather", &wholegraph_torch::gather)
      .op("wholegraph::scatter", &wholegraph_torch::scatter);

