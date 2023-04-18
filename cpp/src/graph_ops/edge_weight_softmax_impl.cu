#include "edge_weight_softmax_func.cuh"
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {
REGISTER_DISPATCH_ONE_TYPE(EdgeWeightSoftmaxForward, edge_weight_softmax_forward_func, HALF_FLOAT)
REGISTER_DISPATCH_ONE_TYPE(EdgeWeightSoftmaxBackward, edge_weight_softmax_backward_func, HALF_FLOAT)

wholememory_error_code_t edge_weight_softmax_forward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_matrix_description_t edge_weight_matrix_desc,
  void* output_edge_weight_ptr,
  wholememory_matrix_description_t output_edge_weight_matrix_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(edge_weight_matrix_desc.dtype,
                      EdgeWeightSoftmaxForward,
                      csr_row_ptr,
                      csr_row_ptr_array_desc,
                      edge_weight_ptr,
                      edge_weight_matrix_desc,
                      output_edge_weight_ptr,
                      output_edge_weight_matrix_desc,
                      stream);

  } catch (const raft::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}
wholememory_error_code_t edge_weight_softmax_backward_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  void* edge_weight_ptr,
  wholememory_matrix_description_t edge_weight_matrix_desc,
  void* grad_edge_weight_softmax_ptr,
  wholememory_matrix_description_t grad_edge_weight_softmax_matrix_desc,
  void* output_grad_edge_weight_ptr,
  wholememory_matrix_description_t output_grad_edge_weight_matrix_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(grad_edge_weight_softmax_matrix_desc.dtype,
                      EdgeWeightSoftmaxBackward,
                      csr_row_ptr,
                      csr_row_ptr_array_desc,
                      edge_weight_ptr,
                      edge_weight_matrix_desc,
                      grad_edge_weight_softmax_ptr,
                      grad_edge_weight_softmax_matrix_desc,
                      output_grad_edge_weight_ptr,
                      output_grad_edge_weight_matrix_desc,
                      stream);

  } catch (const raft::cuda_error& rle) {
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
