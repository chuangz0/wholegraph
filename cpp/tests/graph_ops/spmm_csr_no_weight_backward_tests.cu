#include <gtest/gtest.h>

#include "wholememory/env_func_ptrs.hpp"
#include <wholememory/graph_op.h>
#include <wholememory/tensor_description.h>

#include "../wholegraph_ops/graph_sampling_test_utils.hpp"
#include "./spmm_csr_no_weight_utils.hpp"
#include "error.hpp"

typedef struct SpMMCSRNoWeightBackwardTestParam {
  wholememory_array_description_t get_csr_row_ptr_desc() const
  {
    return wholememory_create_array_desc(target_node_count + 1, 0, WHOLEMEMORY_DT_INT);
  }
  wholememory_array_description_t get_csr_col_ptr_desc() const
  {
    return wholememory_create_array_desc(graph_edge_count, 0, WHOLEMEMORY_DT_INT);
  }

  SpMMCSRNoWeightBackwardTestParam& set_target_node_count(int new_target_node_count)
  {
    target_node_count = new_target_node_count;
    return *this;
  }
  SpMMCSRNoWeightBackwardTestParam& set_total_node_count(int new_total_node_count)
  {
    total_node_count = new_total_node_count;
    return *this;
  }
  SpMMCSRNoWeightBackwardTestParam& set_graph_edge_count(int new_graph_edge_count)
  {
    graph_edge_count = new_graph_edge_count;
    return *this;
  }
  SpMMCSRNoWeightBackwardTestParam& set_feature_dim(int new_feature_dim)
  {
    feature_dim = new_feature_dim;
    return *this;
  }
  SpMMCSRNoWeightBackwardTestParam& set_feature_dtype(wholememory_dtype_t new_feature_dtype)
  {
    feature_dtype = new_feature_dtype;
    return *this;
  }

  SpMMCSRNoWeightBackwardTestParam& set_aggregator(int new_aggregator)
  {
    aggregator = new_aggregator;
    return *this;
  }

  int get_target_node_count() const { return target_node_count; }
  int get_total_node_count() const { return total_node_count; }
  int get_graph_edge_count() const { return graph_edge_count; }
  int get_feature_dim() const { return feature_dim; }
  int get_aggregator() const { return aggregator; }
  wholememory_dtype_t get_feature_dtype() const { return feature_dtype; }
  int target_node_count             = 3;
  int total_node_count              = 10;
  int graph_edge_count              = 20;
  int feature_dim                   = 12;
  int aggregator                    = MEAN_AGGREGATOR;
  wholememory_dtype_t feature_dtype = WHOLEMEMORY_DT_FLOAT;
} SpMMCSRNoWeightBackwardTestParam;

class SpMMCSRNoWeightBackwardParameterTests
  : public ::testing::TestWithParam<SpMMCSRNoWeightBackwardTestParam> {};

TEST_P(SpMMCSRNoWeightBackwardParameterTests, SpMMCSRNoWeightTest)
{
  auto params = GetParam();
  int dev_count;
  EXPECT_EQ(cudaGetDeviceCount(&dev_count), cudaSuccess);

  auto csr_row_ptr_desc  = params.get_csr_row_ptr_desc();
  auto csr_col_ptr_desc  = params.get_csr_col_ptr_desc();
  auto target_node_count = params.get_target_node_count();
  auto total_node_count  = params.get_total_node_count();
  auto feature_dim       = params.get_feature_dim();
  auto graph_edge_count  = params.get_graph_edge_count();
  auto aggregator        = params.get_aggregator();
  auto feature_dtype     = params.get_feature_dtype();

  void* host_csr_row_ptr = (void*)malloc(wholememory_get_memory_size_from_array(&csr_row_ptr_desc));
  void* host_csr_col_ptr = (void*)malloc(wholememory_get_memory_size_from_array(&csr_col_ptr_desc));
  graph_ops::testing::gen_local_csr_graph(target_node_count,
                                          total_node_count,
                                          graph_edge_count,
                                          host_csr_row_ptr,
                                          csr_row_ptr_desc,
                                          host_csr_col_ptr,
                                          csr_col_ptr_desc);
  int64_t input_grad_feature_sizes[2] = {target_node_count, feature_dim};
  auto input_grad_feature_desc =
    wholememory_create_matrix_desc(input_grad_feature_sizes, feature_dim, 0, feature_dtype);
  void* host_input_grad_feature_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&input_grad_feature_desc));
  graph_ops::testing::gen_features(host_input_grad_feature_ptr, input_grad_feature_desc);

  cudaStream_t stream;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  void *host_ref_output_grad_feature = nullptr, *host_output_grad_feature = nullptr;
  void *dev_csr_row_ptr, *dev_csr_col_ptr, *dev_input_grad_feature_ptr,
    *dev_output_grad_feature_ptr;
  int64_t output_sizes[2] = {total_node_count, feature_dim};
  auto output_feature_desc =
    wholememory_create_matrix_desc(output_sizes, feature_dim, 0, feature_dtype);
  EXPECT_EQ(cudaMalloc(&dev_csr_row_ptr, wholememory_get_memory_size_from_array(&csr_row_ptr_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_csr_col_ptr, wholememory_get_memory_size_from_array(&csr_col_ptr_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_input_grad_feature_ptr,
                       wholememory_get_memory_size_from_matrix(&input_grad_feature_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_output_grad_feature_ptr,
                       wholememory_get_memory_size_from_matrix(&output_feature_desc)),
            cudaSuccess);

  EXPECT_EQ(cudaMemcpy(dev_csr_row_ptr,
                       host_csr_row_ptr,
                       wholememory_get_memory_size_from_array(&csr_row_ptr_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_csr_col_ptr,
                       host_csr_col_ptr,
                       wholememory_get_memory_size_from_array(&csr_col_ptr_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_input_grad_feature_ptr,
                       host_input_grad_feature_ptr,
                       wholememory_get_memory_size_from_matrix(&input_grad_feature_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  host_output_grad_feature = malloc(wholememory_get_memory_size_from_matrix(&output_feature_desc));
  host_ref_output_grad_feature =
    malloc(wholememory_get_memory_size_from_matrix(&output_feature_desc));

  wholememory_tensor_description_t csr_row_ptr_tensor_desc, csr_col_ptr_tensor_desc,
    input_grad_feature_tensor_desc, output_grad_feature_tensor_desc;
  wholememory_tensor_t csr_row_ptr_tensor, csr_col_ptr_tensor, input_grad_feature_tensor,
    output_grad_feature_tensor;

  wholememory_copy_array_desc_to_tensor(&csr_row_ptr_tensor_desc, &csr_row_ptr_desc);
  wholememory_copy_array_desc_to_tensor(&csr_col_ptr_tensor_desc, &csr_col_ptr_desc);
  wholememory_copy_matrix_desc_to_tensor(&input_grad_feature_tensor_desc, &input_grad_feature_desc);
  wholememory_copy_matrix_desc_to_tensor(&output_grad_feature_tensor_desc, &output_feature_desc);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_row_ptr_tensor, dev_csr_row_ptr, &csr_row_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_col_ptr_tensor, dev_csr_col_ptr, &csr_col_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(
    wholememory_make_tensor_from_pointer(
      &input_grad_feature_tensor, dev_input_grad_feature_ptr, &input_grad_feature_tensor_desc),
    WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(
    wholememory_make_tensor_from_pointer(
      &output_grad_feature_tensor, dev_output_grad_feature_ptr, &output_grad_feature_tensor_desc),
    WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(spmm_csr_no_weight_backward(csr_row_ptr_tensor,
                                        csr_col_ptr_tensor,
                                        input_grad_feature_tensor,
                                        aggregator,
                                        output_grad_feature_tensor,
                                        stream),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(cudaMemcpyAsync(host_output_grad_feature,
                            dev_output_grad_feature_ptr,
                            wholememory_get_memory_size_from_matrix(&output_feature_desc),
                            cudaMemcpyDeviceToHost,
                            stream),
            cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  graph_ops::testing::host_spmm_csr_no_weight_backward(host_csr_row_ptr,
                                                       csr_row_ptr_desc,
                                                       host_csr_col_ptr,
                                                       csr_col_ptr_desc,
                                                       host_input_grad_feature_ptr,
                                                       input_grad_feature_desc,
                                                       aggregator,
                                                       host_ref_output_grad_feature,
                                                       output_feature_desc);

  graph_ops::testing::host_check_float_matrix_same(host_output_grad_feature,
                                                   output_feature_desc,
                                                   host_ref_output_grad_feature,
                                                   output_feature_desc);

  EXPECT_EQ(cudaFree(dev_csr_row_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_csr_col_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_input_grad_feature_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_grad_feature_ptr), cudaSuccess);

  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_csr_col_ptr != nullptr) free(host_csr_col_ptr);
  if (host_input_grad_feature_ptr != nullptr) free(host_input_grad_feature_ptr);
  if (host_output_grad_feature != nullptr) free(host_output_grad_feature);
  if (host_ref_output_grad_feature != nullptr) free(host_ref_output_grad_feature);

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
}

INSTANTIATE_TEST_SUITE_P(SpMMCSRNoWeightBackwardTests,
                         SpMMCSRNoWeightBackwardParameterTests,
                         ::testing::Values(SpMMCSRNoWeightBackwardTestParam()
                                             .set_target_node_count(512)
                                             .set_total_node_count(1024)
                                             .set_graph_edge_count(3457)
                                             .set_feature_dim(128)
                                             .set_feature_dtype(WHOLEMEMORY_DT_FLOAT)
                                             .set_aggregator(SUM_AGGREGATOR),
                                           SpMMCSRNoWeightBackwardTestParam()
                                             .set_target_node_count(512)
                                             .set_total_node_count(1024)
                                             .set_graph_edge_count(4579)
                                             .set_feature_dim(128)
                                             .set_feature_dtype(WHOLEMEMORY_DT_FLOAT)
                                             .set_aggregator(MEAN_AGGREGATOR),
                                           SpMMCSRNoWeightBackwardTestParam()
                                             .set_target_node_count(368)
                                             .set_total_node_count(3057)
                                             .set_graph_edge_count(2069)
                                             .set_feature_dim(235)
                                             .set_feature_dtype(WHOLEMEMORY_DT_FLOAT)
                                             .set_aggregator(GCN_AGGREGATOR)));