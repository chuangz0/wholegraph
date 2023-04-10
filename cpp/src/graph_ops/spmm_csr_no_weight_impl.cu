#include <cuda_runtime_api.h>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include "spmm_csr_no_weight_func.cuh"
#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

namespace graph_ops {

REGISTER_DISPATCH_ONE_TYPE(SpMMCSRNoWeightForward, spmm_csr_no_weight_forward_func, HALF_FLOAT)

wholememory_error_code_t spmm_csr_no_weight_forward_mapped(
  void* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* feature_ptr,
  wholememory_matrix_description_t feature_desc,
  int aggregator,
  void* output_ptr,
  wholememory_matrix_description_t output_desc,
  cudaStream_t stream)
{
  try {
    DISPATCH_ONE_TYPE(feature_desc.dtype,
                      SpMMCSRNoWeightForward,
                      csr_row_ptr,
                      csr_row_ptr_desc,
                      csr_col_ptr,
                      csr_col_ptr_desc,
                      feature_ptr,
                      feature_desc,
                      aggregator,
                      output_ptr,
                      output_desc,
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
