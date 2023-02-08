#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include <wholememory/wholememory_op.h>

#include "../torch_utils.h"

namespace wholegraph_torch {

torch::Tensor wholememory_gather(int64_t wholememory_handle,
                                 const torch::Tensor& indices,
                                 torch::optional<torch::ScalarType> output_type,
                                 torch::optional<bool> requires_grad) {
  TORCH_CHECK(indices.dim() == 1, "indices should be 1-dim tensor.");
  TORCH_CHECK(indices.dtype() == torch::kInt32 || indices.dtype() == torch::kInt64,
              "indices should be IntTensor or LongTensor.");
  TORCH_CHECK(false, "Not implemented.");
  return torch::Tensor();
}

}  // namespace wholegraph_torch