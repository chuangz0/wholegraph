#include <wholememory/graph_op.h>
#include <graph_ops/spmm_csr_no_weight_impl.h>
#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t spmm_csr_no_weight_forward(wholememory_tensor_t csr_row_ptr_tensor,
                                                    wholememory_tensor_t csr_col_ptr_tensor,
                                                    wholememory_tensor_t feature_tensor,
                                                    int64_t aggregator,
                                                    wholememory_tensor_t output_tensor,
                                                    wholememory_env_func_t * p_env_fns, 
                                                    void* stream) {
  wholememory_tensor_description_t csr_row_ptr_tensor_desc = *wholememory_tensor_get_tensor_description(csr_row_ptr_tensor);
  wholememory_tensor_description_t csr_col_ptr_tensor_desc = *wholememory_tensor_get_tensor_description(csr_col_ptr_tensor);
  wholememory_tensor_description_t feature_tensor_desc = *wholememory_tensor_get_tensor_description(feature_tensor);
  wholememory_tensor_description_t output_tensor_desc = *wholememory_tensor_get_tensor_description(output_tensor);
  if (csr_row_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (feature_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Input feature_tensor should be 2D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_tensor_desc.dim != 2) {
    WHOLEMEMORY_ERROR("Output output_tensor should be 2D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  wholememory_array_description_t csr_row_ptr_desc, csr_col_ptr_desc;
  if (!wholememory_convert_tensor_desc_to_array(&csr_row_ptr_desc,
                                                &csr_row_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  if (!wholememory_convert_tensor_desc_to_array(&csr_col_ptr_desc,
                                                &csr_col_ptr_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  wholememory_matrix_description_t feature_desc, output_desc;
  if (!wholememory_convert_tensor_desc_to_matrix(&feature_desc, &feature_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input feature_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&output_desc, &output_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* csr_row_ptr = wholememory_tensor_get_data_pointer(csr_row_ptr_tensor);
  void* csr_col_ptr = wholememory_tensor_get_data_pointer(csr_col_ptr_tensor);
  void* feature_ptr = wholememory_tensor_get_data_pointer(feature_tensor);
  void* output_ptr  = wholememory_tensor_get_data_pointer(output_tensor);

  return graph_ops::spmm_csr_no_weight_forward_mapped(
    csr_row_ptr,
    csr_row_ptr_desc,
    csr_col_ptr,
    csr_col_ptr_desc,
    feature_ptr,
    feature_desc,
    aggregator,
    output_ptr,
    output_desc,
    p_env_fns,
    static_cast<cudaStream_t>(stream)
  );
}