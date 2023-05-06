#include "gspmm_csr_weighted_func.cuh"
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {
REGISTER_DISPATCH_ONE_TYPE(gSpmmCSRWeightedForward, gspmm_csr_weighted_forward_func, HALF_FLOAT)
REGISTER_DISPATCH_ONE_TYPE(gSpmmCSRWeightedBackward, gspmm_csr_weighted_backward_func, HALF_FLOAT)

wholememory_error_code_t gspmm_csr_weighted_forward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* output_feature_ptr,
  wholememory_tensor_description_t output_feature_tensor_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(feature_tensor_desc.dtype,
                      gSpmmCSRWeightedForward,
                      csr_row_ptr,
                      csr_row_ptr_array_desc,
                      csr_col_ptr,
                      csr_col_ptr_array_desc,
                      edge_weight_ptr,
                      edge_weight_tensor_desc,
                      feature_ptr,
                      feature_tensor_desc,
                      output_feature_ptr,
                      output_feature_tensor_desc,
                      stream);

  } catch (const wholememory::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t gspmm_csr_weighted_backward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_tensor_description_t edge_weight_tensor_desc,
  void* feature_ptr,
  wholememory_tensor_description_t feature_tensor_desc,
  void* grad_feature_ptr,
  wholememory_tensor_description_t grad_feature_tensor,
  void* output_grad_edge_weight_ptr,
  wholememory_tensor_description_t output_grad_edge_weight_tensor_desc,
  void* output_grad_feature_ptr,
  wholememory_tensor_description_t output_grad_feature_tensor_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(feature_tensor_desc.dtype,
                      gSpmmCSRWeightedBackward,
                      csr_row_ptr,
                      csr_row_ptr_array_desc,
                      csr_col_ptr,
                      csr_col_ptr_array_desc,
                      edge_weight_ptr,
                      edge_weight_tensor_desc,
                      feature_ptr,
                      feature_tensor_desc,
                      grad_feature_ptr,
                      grad_feature_tensor,
                      output_grad_edge_weight_ptr,
                      output_grad_edge_weight_tensor_desc,
                      output_grad_feature_ptr,
                      output_grad_feature_tensor_desc,
                      stream);

  } catch (const wholememory::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace graph_ops
