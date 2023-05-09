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
#include <error.hpp>
#include <gtest/gtest.h>

#include "./spmm_csr_no_weight_utils.hpp"
#include "spadd_gat_csr_utils.hpp"
#include "wholememory/initialize.hpp"
#include <wholememory/graph_op.h>
#include <wholememory/tensor_description.h>

typedef struct GraphSpADDGATForwardTestParam {
  GraphSpADDGATForwardTestParam& set_graph_row_num(int new_graph_row_num)
  {
    graph_row_num = new_graph_row_num;
    return *this;
  }
  GraphSpADDGATForwardTestParam& set_graph_col_num(int new_graph_col_num)
  {
    graph_col_num = new_graph_col_num;
    return *this;
  }
  GraphSpADDGATForwardTestParam& set_graph_edge_num(int new_graph_edge_num)
  {
    graph_edge_num = new_graph_edge_num;
    return *this;
  }
  GraphSpADDGATForwardTestParam& set_num_heads(int new_num_heads)
  {
    num_heads = new_num_heads;
    return *this;
  }
  GraphSpADDGATForwardTestParam& set_weight_dtype(wholememory_dtype_t new_weight_dtype)
  {
    weight_dtype = new_weight_dtype;
    return *this;
  }

  wholememory_array_description_t get_csr_row_ptr_array_desc()
  {
    return wholememory_create_array_desc(graph_row_num + 1, 0, WHOLEMEMORY_DT_INT);
  }
  wholememory_array_description_t get_csr_col_ptr_array_desc()
  {
    return wholememory_create_array_desc(graph_edge_num, 0, WHOLEMEMORY_DT_INT);
  }

  wholememory_matrix_description_t get_weight_left_matrix_desc() const
  {
    int64_t sizes[2] = {graph_row_num, num_heads};
    return wholememory_create_matrix_desc(sizes, num_heads, 0, weight_dtype);
  }
  wholememory_matrix_description_t get_weight_right_matrix_desc() const
  {
    int64_t sizes[2] = {graph_col_num, num_heads};
    return wholememory_create_matrix_desc(sizes, num_heads, 0, weight_dtype);
  }

  int get_graph_row_num() const { return graph_row_num; }
  int get_graph_col_num() const { return graph_col_num; }
  int get_graph_edge_num() const { return graph_edge_num; }
  int get_num_heads() const { return num_heads; }
  wholememory_dtype_t get_weight_dtype() const { return weight_dtype; }

  int graph_row_num                = 3;
  int graph_col_num                = 4;
  int graph_edge_num               = 10;
  int num_heads                    = 8;
  wholememory_dtype_t weight_dtype = WHOLEMEMORY_DT_FLOAT;

} GraphSpADDGATForwardTestParam;

class GraphSpADDGATForwardParameterTests
  : public ::testing::TestWithParam<GraphSpADDGATForwardTestParam> {};

TEST_P(GraphSpADDGATForwardParameterTests, SpADDGATForwardParameterTest)
{
  auto params = GetParam();
  int dev_count;
  EXPECT_EQ(cudaGetDeviceCount(&dev_count), cudaSuccess);
  EXPECT_GE(dev_count, 1);

  cudaStream_t stream;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  auto graph_row_num            = params.get_graph_row_num();
  auto graph_col_num            = params.get_graph_col_num();
  auto graph_edge_num           = params.get_graph_edge_num();
  auto num_heads                = params.get_num_heads();
  auto csr_row_ptr_array_desc   = params.get_csr_row_ptr_array_desc();
  auto csr_col_ptr_array_desc   = params.get_csr_col_ptr_array_desc();
  auto weight_left_matrix_desc  = params.get_weight_left_matrix_desc();
  auto weight_right_matrix_desc = params.get_weight_right_matrix_desc();

  void* host_csr_row_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc));
  void* host_csr_col_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc));
  graph_ops::testing::gen_local_csr_graph(graph_row_num,
                                          graph_col_num,
                                          graph_edge_num,
                                          host_csr_row_ptr,
                                          csr_row_ptr_array_desc,
                                          host_csr_col_ptr,
                                          csr_col_ptr_array_desc);

  void* host_weight_left_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&weight_left_matrix_desc));
  void* host_weight_right_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&weight_right_matrix_desc));
  graph_ops::testing::gen_features(host_weight_left_ptr, weight_left_matrix_desc);
  graph_ops::testing::gen_features(host_weight_right_ptr, weight_right_matrix_desc);

  void *dev_csr_row_ptr, *dev_csr_col_ptr, *dev_weight_left_ptr, *dev_weight_right_ptr,
    *dev_output_score;
  int64_t output_sizes[2] = {graph_edge_num, num_heads};
  auto output_score_matrix_desc =
    wholememory_create_matrix_desc(output_sizes, num_heads, 0, weight_left_matrix_desc.dtype);
  EXPECT_EQ(
    cudaMalloc(&dev_csr_row_ptr, wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(
    cudaMalloc(&dev_csr_col_ptr, wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc)),
    cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_weight_left_ptr,
                       wholememory_get_memory_size_from_matrix(&weight_left_matrix_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_weight_right_ptr,
                       wholememory_get_memory_size_from_matrix(&weight_right_matrix_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMalloc(&dev_output_score,
                       wholememory_get_memory_size_from_matrix(&output_score_matrix_desc)),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_csr_row_ptr,
                       host_csr_row_ptr,
                       wholememory_get_memory_size_from_array(&csr_row_ptr_array_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  EXPECT_EQ(cudaMemcpy(dev_csr_col_ptr,
                       host_csr_col_ptr,
                       wholememory_get_memory_size_from_array(&csr_col_ptr_array_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_weight_left_ptr,
                       host_weight_left_ptr,
                       wholememory_get_memory_size_from_matrix(&weight_left_matrix_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev_weight_right_ptr,
                       host_weight_right_ptr,
                       wholememory_get_memory_size_from_matrix(&weight_right_matrix_desc),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  wholememory_tensor_description_t csr_row_ptr_tensor_desc, csr_col_ptr_tensor_desc,
    weight_left_tensor_desc, weight_right_tensor_desc, output_score_tensor_desc;
  wholememory_tensor_t csr_row_ptr_tensor, csr_col_ptr_tensor, weight_left_tensor,
    weight_right_tensor, output_score_tensor;
  wholememory_copy_array_desc_to_tensor(&csr_row_ptr_tensor_desc, &csr_row_ptr_array_desc);
  wholememory_copy_array_desc_to_tensor(&csr_col_ptr_tensor_desc, &csr_col_ptr_array_desc);
  wholememory_copy_matrix_desc_to_tensor(&weight_left_tensor_desc, &weight_left_matrix_desc);
  wholememory_copy_matrix_desc_to_tensor(&weight_right_tensor_desc, &weight_right_matrix_desc);
  wholememory_copy_matrix_desc_to_tensor(&output_score_tensor_desc, &output_score_matrix_desc);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_row_ptr_tensor, dev_csr_row_ptr, &csr_row_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &csr_col_ptr_tensor, dev_csr_col_ptr, &csr_col_ptr_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &weight_left_tensor, dev_weight_left_ptr, &weight_left_tensor_desc),
            WHOLEMEMORY_SUCCESS);
  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &weight_right_tensor, dev_weight_right_ptr, &weight_right_tensor_desc),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(wholememory_make_tensor_from_pointer(
              &output_score_tensor, dev_output_score, &output_score_tensor_desc),
            WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(spadd_gat_csr_foward(csr_row_ptr_tensor,
                                 csr_col_ptr_tensor,
                                 weight_left_tensor,
                                 weight_right_tensor,
                                 output_score_tensor,
                                 stream),
            WHOLEMEMORY_SUCCESS);

  void* host_output_score_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&output_score_matrix_desc));
  void* host_ref_output_score_ptr =
    (void*)malloc(wholememory_get_memory_size_from_matrix(&output_score_matrix_desc));

  EXPECT_EQ(cudaMemcpyAsync(host_output_score_ptr,
                            dev_output_score,
                            wholememory_get_memory_size_from_matrix(&output_score_matrix_desc),
                            cudaMemcpyDeviceToHost,
                            stream),
            cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  graph_ops::testing::host_spadd_gat_csr_forward(host_csr_row_ptr,
                                                 csr_row_ptr_array_desc,
                                                 host_csr_col_ptr,
                                                 csr_col_ptr_array_desc,
                                                 host_weight_left_ptr,
                                                 weight_left_matrix_desc,
                                                 host_weight_right_ptr,
                                                 weight_right_matrix_desc,
                                                 host_ref_output_score_ptr,
                                                 output_score_matrix_desc);

  graph_ops::testing::host_check_float_matrix_same(host_output_score_ptr,
                                                   output_score_matrix_desc,
                                                   host_ref_output_score_ptr,
                                                   output_score_matrix_desc);

  EXPECT_EQ(cudaFree(dev_csr_row_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_csr_col_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_weight_left_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_weight_right_ptr), cudaSuccess);
  EXPECT_EQ(cudaFree(dev_output_score), cudaSuccess);

  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_csr_col_ptr != nullptr) free(host_csr_col_ptr);
  if (host_weight_left_ptr != nullptr) free(host_weight_left_ptr);
  if (host_weight_right_ptr != nullptr) free(host_weight_right_ptr);
  if (host_output_score_ptr != nullptr) free(host_output_score_ptr);
  if (host_ref_output_score_ptr != nullptr) free(host_ref_output_score_ptr);

  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
}

INSTANTIATE_TEST_SUITE_P(SpADDGATForwardOpTests,
                         GraphSpADDGATForwardParameterTests,
                         ::testing::Values(GraphSpADDGATForwardTestParam()
                                             .set_graph_row_num(1025)
                                             .set_graph_col_num(2379)
                                             .set_graph_edge_num(10793)
                                             .set_num_heads(7)));
