/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <algorithm>
#include <cstdint>
#include <getopt.h>
#include <random>
#include <sys/time.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <string_view>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_op.h>

#include "../common/wholegraph_benchmark.hpp"
#include "old_gather.cuh"
#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/env_func_ptrs.hpp"
#include "wholememory/initialize.hpp"

#include "../../tests/wholememory/wholememory_test_utils.hpp"
#include "../../tests/wholememory_ops/embedding_test_utils.hpp"

namespace wholegraph::bench::gather_scatter {

enum class GatherMode {
  kSingleReadSingle = 0,
  kSingleReadAllExcludeSelf,
  kAllExcludeSelfReadSingle,
  kAllReadAll,
  kGatherModeNum
};

struct SmGatherParam {
  int64_t embedding_table_size = 1024L * 1024L * 1024L * 40;  // 40M * 128*bytes
  int64_t embedding_dim        = 128;
  int64_t embedding_stride     = 128;
  int64_t output_stride        = 128;

  int64_t block_min      = 4;
  int64_t block_step     = 4;
  int64_t block_max      = 1024;
  int64_t gather_size    = 1024L * 1024L * 1024 * 4;  // 4G
  int loop_count         = 10;
  GatherMode gather_mode = GatherMode::kAllReadAll;

  wholememory_dtype_t embedding_type            = WHOLEMEMORY_DT_FLOAT;
  wholememory_dtype_t indices_type              = WHOLEMEMORY_DT_INT64;
  wholememory_dtype_t output_type               = WHOLEMEMORY_DT_FLOAT;
  int64_t embedding_storage_offset              = 0;
  int64_t indices_storage_offset                = 0;
  int64_t output_storage_offset                 = 0;
  wholememory_memory_type_t memory_type         = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location = WHOLEMEMORY_ML_DEVICE;
  std::string server_addr                       = "localhost";
  int server_port                               = 24987;
  int node_rank                                 = 0;
  int node_size                                 = 1;
  int num_gpu                                   = 0;
  bool check_result                             = true;

  std::string get_server_addr() const { return server_addr; }
  int get_server_port() const { return server_port; }
  int get_node_rank() const { return node_rank; }
  int get_node_size() const { return node_size; }
  [[nodiscard]] wholememory_matrix_description_t get_embedding_desc() const
  {
    int64_t embedding_entry_count = get_embedding_entry_count();
    int64_t matrix_sizes[2]       = {embedding_entry_count, embedding_dim};
    return wholememory_create_matrix_desc(
      matrix_sizes, embedding_stride, embedding_storage_offset, embedding_type);
  }

  [[nodiscard]] int64_t get_embedding_entry_count() const
  {
    return embedding_table_size / wholememory_dtype_get_element_size(embedding_type) /
           embedding_dim;
  }

  [[nodiscard]] int64_t get_embedding_dim() const { return embedding_dim; }
  [[nodiscard]] int64_t get_embedding_stride() const { return embedding_stride; }
  [[nodiscard]] int64_t get_embedding_storage_offset() const { return embedding_storage_offset; }
  [[nodiscard]] wholememory_dtype_t get_embedding_type() const { return embedding_type; }
  [[nodiscard]] int64_t get_embedding_table_size() const { return embedding_table_size; }

  [[nodiscard]] int64_t get_indices_count() const
  {
    return gather_size / wholememory_dtype_get_element_size(embedding_type) / embedding_dim;
  }
  [[nodiscard]] wholememory_array_description_t get_indices_desc() const
  {
    int64_t indices_count = get_indices_count();
    return wholememory_create_array_desc(indices_count, indices_storage_offset, indices_type);
  }
  [[nodiscard]] wholememory_matrix_description_t get_output_desc() const
  {
    int64_t indices_count   = get_indices_count();
    int64_t output_sizes[2] = {indices_count, embedding_dim};
    return wholememory_create_matrix_desc(
      output_sizes, output_stride, output_storage_offset, output_type);
  }

  [[nodiscard]] int64_t get_embedding_granularity() const
  {
    return embedding_stride * wholememory_dtype_get_element_size(embedding_type);
  }

  [[nodiscard]] int get_num_gpu() const { return num_gpu; }
  [[nodiscard]] bool need_check_result() const { return check_result; }
  [[nodiscard]] wholememory_memory_type_t get_memory_type() const { return memory_type; }

  [[nodiscard]] wholememory_memory_location_t get_memory_location() const
  {
    return memory_location;
  }
  [[nodiscard]] int64_t get_gather_size() const { return gather_size; }

  SmGatherParam& set_memory_type(wholememory_memory_type_t new_memory_type)
  {
    memory_type = new_memory_type;
    return *this;
  }
  SmGatherParam& set_embedding_storage_offset(int64_t new_embedding_storage_offset)
  {
    embedding_storage_offset = new_embedding_storage_offset;
    return *this;
  }
  SmGatherParam& set_indices_storage_offset(int64_t new_indices_storage_offset)
  {
    indices_storage_offset = new_indices_storage_offset;
    return *this;
  }
  SmGatherParam& set_output_storage_offset(int64_t new_output_storage_offset)
  {
    output_storage_offset = new_output_storage_offset;
    return *this;
  }
  SmGatherParam& set_indices_type(wholememory_dtype_t new_indices_type)
  {
    indices_type = new_indices_type;
    return *this;
  }
  SmGatherParam& set_embedding_type(wholememory_dtype_t new_embedding_type)
  {
    embedding_type = new_embedding_type;
    return *this;
  }
  SmGatherParam& set_output_type(wholememory_dtype_t new_output_type)
  {
    output_type = new_output_type;
    return *this;
  }
  SmGatherParam& set_gather_size(int64_t new_gather_size)
  {
    gather_size = new_gather_size;
    return *this;
  }
  SmGatherParam& set_block_min(int64_t new_block_min)
  {
    block_min = new_block_min;
    return *this;
  }
  SmGatherParam& set_block_max(int64_t new_block_max)
  {
    block_max = new_block_max;
    return *this;
  }
  SmGatherParam& set_block_step(int64_t new_block_step)
  {
    block_step = new_block_step;
    return *this;
  }
  SmGatherParam& set_loop_count(int64_t new_loop_count)
  {
    loop_count = new_loop_count;
    return *this;
  }
  SmGatherParam& set_num_gpu(int64_t new_num_gpu)
  {
    num_gpu = new_num_gpu;
    return *this;
  }

  SmGatherParam& set_need_check_result(bool new_need_check_result)
  {
    check_result = new_need_check_result;
    return *this;
  }

  SmGatherParam& set_server_port(int new_server_port)
  {
    server_port = new_server_port;
    return *this;
  }
  SmGatherParam& set_server_addr(std::string new_server_addr)
  {
    server_addr = new_server_addr;
    return *this;
  }
  SmGatherParam& set_node_rank(int new_node_rank)
  {
    node_rank = new_node_rank;
    return *this;
  }

  SmGatherParam& set_node_size(int new_node_size)
  {
    node_size = new_node_size;
    return *this;
  }

  SmGatherParam& set_num_gpu(int new_num_gpu)
  {
    num_gpu = new_num_gpu;
    return *this;
  }

  SmGatherParam& set_embedding_dim(int64_t new_embedding_dim)
  {
    embedding_dim    = new_embedding_dim;
    embedding_stride = new_embedding_dim;
    output_stride    = new_embedding_dim;
    return *this;
  }

  SmGatherParam& set_memory_location(wholememory_memory_location_t new_memory_location)
  {
    memory_location = new_memory_location;
    return *this;
  }
  SmGatherParam& set_embedding_table_size(int64_t new_embedding_table_size)
  {
    embedding_table_size = new_embedding_table_size;
    return *this;
  }

  // SmGatherParam& set_embedding_dim(int64_t new_embedding_dim){
  //   embedding_dim = new_embedding_dim;
  //   return *this;
  // }
  SmGatherParam& set_gather_mode(GatherMode new_gather_mode)
  {
    gather_mode = new_gather_mode;
    return *this;
  }
  [[nodiscard]] GatherMode get_gather_mode() const { return gather_mode; }
};

template <typename T>
std::vector<T> get_possible_indices(int64_t max_indices,
                                    int rank,
                                    int all_rank,
                                    SmGatherParam& params)
{
  std::vector<T> indices;
  indices.reserve(max_indices);
  int64_t average_indices_per_rank = max_indices / all_rank;
  // std::vector<T> average_indices;
  // average_indices.reserve(average_indices_per_rank);
  // for(int i=0;i<average_indices_per_rank;i++){
  //     average_indices.push_back(i);
  // }
  switch (params.get_gather_mode()) {
    case GatherMode::kSingleReadSingle: {
      for (int i = 0; i < average_indices_per_rank; i++) {
        indices.push_back(i + average_indices_per_rank);
      }
    } break;

      // 7 read 1
    case GatherMode::kAllExcludeSelfReadSingle: {
      for (int i = 0; i < average_indices_per_rank; i++) {
        indices.push_back(i);
      }

    } break;

      // 1 read 7
    case GatherMode::kSingleReadAllExcludeSelf: {
      for (int rank_i = 0; rank_i < all_rank; rank_i++) {
        if (rank_i != rank) {
          for (int i = 0; i < average_indices_per_rank; i++) {
            indices.push_back(i + average_indices_per_rank * rank_i);
          }
        }
      }
    } break;

      // 8 read 8
    case GatherMode::kAllReadAll: {
      for (int rank_i = 0; rank_i < all_rank; rank_i++) {
        for (int i = 0; i < average_indices_per_rank; i++) {
          indices.push_back(i + average_indices_per_rank * rank_i);
        }
      }
    } break;
    default: break;
  }

  return indices;
}
template <typename IndexT>
void host_get_random_integer_indices(void* indices,
                                     wholememory_array_description_t indices_desc,
                                     int64_t max_indices,
                                     SmGatherParam& params,
                                     wholememory_comm_t comm)
{
  std::vector<IndexT> possible_indices =
    get_possible_indices<IndexT>(max_indices, comm->world_rank, comm->world_size, params);
  IndexT* indices_ptr = static_cast<IndexT*>(indices);
  if (possible_indices.size() >= indices_desc.size) {
    std::sample(possible_indices.begin(),
                possible_indices.end(),
                indices_ptr,
                indices_desc.size,
                std::mt19937{std::random_device{}()});
  } else {
    int64_t remaing      = indices_desc.size;
    IndexT* indice_alter = indices_ptr;
    while (remaing > 0) {
      int64_t sample_count =
        (possible_indices.size() > remaing) ? remaing : possible_indices.size();

      std::sample(possible_indices.begin(),
                  possible_indices.end(),
                  indice_alter,
                  sample_count,
                  std::mt19937{std::random_device{}()});
      indice_alter += sample_count;
      remaing -= sample_count;
    }
  }
  std::shuffle(indices_ptr, indices_ptr + indices_desc.size, std::mt19937(std::random_device{}()));
}

void host_random_init_integer_indices_with_gahter_mode(void* indices,
                                                       wholememory_array_description_t indices_desc,
                                                       int64_t max_indices,
                                                       SmGatherParam& params,
                                                       wholememory_comm_t comm)
{
  if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_random_integer_indices<int>(indices, indices_desc, max_indices, params, comm);
  } else {
    host_get_random_integer_indices<int64_t>(indices, indices_desc, max_indices, params, comm);
  }
}

/*
benchmark
1.1个src 7个dst
2.1个src 1个dst
3.7个src 1个dst
4. 8个src 8个dst
*/

/**
 * enum class GatherMode {
  kSingleReadSingle = 0,
  kSingleReadAllExcludeSelf,
  kAllExcludeSelfReadSingle,
  kAllReadAll
};

*/
bool select_this_rank(SmGatherParam& params, int local_rank, int local_size)
{
  // 1read 1
  switch (params.get_gather_mode()) {
    case GatherMode::kSingleReadSingle: {
      return local_rank == 0;
    } break;

      // 7 read 1
    case GatherMode::kAllExcludeSelfReadSingle: {
      return local_rank != 0;
    } break;

    case GatherMode::kSingleReadAllExcludeSelf: {
      return local_rank == 0;
    } break;

    case GatherMode::kAllReadAll: {
      return true;
    } break;
    default: break;
  }
  return false;
}

wholememory_error_code_t gather_fun_new_or_old(wholememory_gref_t embedding_gref,
                                               wholememory_matrix_description_t embedding_desc,
                                               void* indices,
                                               int64_t indice_count,
                                               void* output,
                                               wholememory_matrix_description_t output_desc,
                                               int64_t block_count,
                                               bool use_old,
                                               cudaStream_t stream)
{
  // printf("***** run in gather_new_or _old\n");
  WHOLEMEMORY_CHECK(embedding_desc.dtype == WHOLEMEMORY_DT_FLOAT);
  WHOLEMEMORY_CHECK(output_desc.dtype == WHOLEMEMORY_DT_FLOAT);
  // printf("***** run after gather_new_or _old\n");

  if (use_old) {
    wholememory_ops::old_gather_temp_func<float, int64_t, float>(embedding_gref,
                                                                 embedding_desc,
                                                                 indices,
                                                                 indice_count,
                                                                 output,
                                                                 output_desc,
                                                                 block_count,
                                                                 stream);
  } else {
    wholememory_ops::new_gather_temp_func<float, int64_t, float>(embedding_gref,
                                                                 embedding_desc,
                                                                 indices,
                                                                 indice_count,
                                                                 output,
                                                                 output_desc,
                                                                 block_count,
                                                                 stream);
  }

  // printf("***** run finish gather_new_or _old\n");

  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t gather_fun_with_block_count(wholememory_tensor_t wholememory_tensor,
                                                     wholememory_tensor_t indices_tensor,
                                                     wholememory_tensor_t output_tensor,
                                                     int64_t block_count,
                                                     bool use_old,
                                                     cudaStream_t stream)
{
  // return WHOLEMEMORY_SUCCESS;

  wholememory_gref_t gref;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(wholememory_tensor, &gref));
  // WHOLEMEMORY_CHECK_NOTHROW(indices_tensor.dtype == WHOLEMEMORY_DT_INT64);
  // wholememory_tensor_get_tensor_descript

  wholememory_matrix_description_t matrix_description;
  auto tensor_description = *wholememory_tensor_get_tensor_description(wholememory_tensor);
  if (!wholememory_convert_tensor_desc_to_matrix(&matrix_description, &tensor_description)) {
    WHOLEMEMORY_ERROR("Input wholememory_tensor convert to matrix failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  void* indices = wholememory_tensor_get_data_pointer(indices_tensor);
  void* output  = wholememory_tensor_get_data_pointer(output_tensor);
  wholememory_array_description_t indices_desc;
  wholememory_matrix_description_t output_desc;
  if (!wholememory_convert_tensor_desc_to_array(
        &indices_desc, wholememory_tensor_get_tensor_description(indices_tensor))) {
    WHOLEMEMORY_ERROR("Convert indices tensor to array failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(
        &output_desc, wholememory_tensor_get_tensor_description(output_tensor))) {
    WHOLEMEMORY_ERROR("Convert output tensor to matrix failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  auto ret = gather_fun_new_or_old(gref,
                                   matrix_description,
                                   indices,
                                   indices_desc.size,
                                   output,
                                   output_desc,
                                   block_count,
                                   use_old,
                                   stream);

  return ret;
}

void sm_count_gather_benchmark(SmGatherParam& params)
{
  int g_dev_count = ForkGetDeviceCount();
  WHOLEMEMORY_CHECK_NOTHROW(g_dev_count >= 1);
  if (params.get_num_gpu() == 0 || params.get_num_gpu() > g_dev_count) {
    params.set_num_gpu(g_dev_count);
  }
  MultiProcessRun(
    params.get_num_gpu(),
    [&params](int local_rank, int local_size) {
      WHOLEMEMORY_CHECK_NOTHROW(wholememory_init(0) == WHOLEMEMORY_SUCCESS);
      WM_CUDA_CHECK_NO_THROW(cudaSetDevice(local_rank));
      int world_size = local_size * params.get_node_size();
      int world_rank = params.get_node_rank() * params.get_num_gpu() + local_rank;

      SideBandCommunicator* side_band_communicator = StartSidebandCommunicator(
        world_rank, world_size, params.get_server_addr().c_str(), params.get_server_port());

      wholememory_comm_t wm_comm =
        create_communicator_by_socket(side_band_communicator, world_rank, world_size);

      ShutDownSidebandCommunicator(side_band_communicator);

      auto embedding_desc         = params.get_embedding_desc();
      auto indices_desc           = params.get_indices_desc();
      auto output_desc            = params.get_output_desc();
      size_t embedding_entry_size = params.get_embedding_granularity();

      wholememory_tensor_t embedding_tensor;
      wholememory_tensor_description_t embedding_tensor_desc;
      wholememory_copy_matrix_desc_to_tensor(&embedding_tensor_desc, &embedding_desc);
      WHOLEMEMORY_CHECK_NOTHROW(wholememory_create_tensor(&embedding_tensor,
                                                          &embedding_tensor_desc,
                                                          wm_comm,
                                                          params.get_memory_type(),
                                                          params.get_memory_location()) ==
                                WHOLEMEMORY_SUCCESS);

      cudaStream_t stream;
      WM_CUDA_CHECK_NO_THROW(cudaStreamCreate(&stream));
      wholememory_handle_t embedding_handle =
        wholememory_tensor_get_memory_handle(embedding_tensor);
      wholememory_ops::testing::device_random_init_local_embedding_table(
        embedding_handle, embedding_desc, stream);

      WHOLEMEMORY_CHECK_NOTHROW(wholememory_communicator_barrier(wm_comm) == WHOLEMEMORY_SUCCESS);
      wholememory_tensor_t indices_tensor, output_tensor;
      void *dev_indices = nullptr, *dev_gather_buffer = nullptr;
      void* host_indices                 = nullptr;
      const bool this_rank_been_selected = select_this_rank(params, local_rank, local_size);
      if (this_rank_been_selected) {
        size_t gather_buffer_size  = params.get_gather_size();
        size_t indices_buffer_size = wholememory_get_memory_size_from_array(&indices_desc);

        WM_CUDA_CHECK_NO_THROW(cudaMallocHost(&host_indices, indices_buffer_size));
        WM_CUDA_CHECK_NO_THROW(cudaMalloc(&dev_indices, indices_buffer_size));
        WM_CUDA_CHECK_NO_THROW(cudaMalloc(&dev_gather_buffer, gather_buffer_size));

        host_random_init_integer_indices_with_gahter_mode(
          host_indices, indices_desc, embedding_desc.sizes[0], params, wm_comm);
        WM_CUDA_CHECK_NO_THROW(
          cudaMemcpyAsync(dev_indices,
                          host_indices,
                          wholememory_get_memory_size_from_array(&indices_desc),
                          cudaMemcpyHostToDevice,
                          stream));
        WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
        wholememory_tensor_description_t indices_tensor_desc, output_tensor_desc;
        wholememory_copy_array_desc_to_tensor(&indices_tensor_desc, &indices_desc);
        wholememory_copy_matrix_desc_to_tensor(&output_tensor_desc, &output_desc);
        WHOLEMEMORY_CHECK_NOTHROW(wholememory_make_tensor_from_pointer(
                                    &indices_tensor, dev_indices, &indices_tensor_desc) ==
                                  WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK_NOTHROW(wholememory_make_tensor_from_pointer(
                                    &output_tensor, dev_gather_buffer, &output_tensor_desc) ==
                                  WHOLEMEMORY_SUCCESS);
        WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));

        WHOLEMEMORY_CHECK_NOTHROW(gather_fun_with_block_count(embedding_tensor,
                                                              indices_tensor,
                                                              output_tensor,
                                                              /*block_count*/ 1024,
                                                              /*use_old*/ true,
                                                              stream) == WHOLEMEMORY_SUCCESS);

        WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));

        if (params.check_result) {
          void* dev_reference_buffer = nullptr;
          WM_CUDA_CHECK_NO_THROW(cudaMalloc(&dev_reference_buffer, gather_buffer_size));

          wholememory_ops::testing::device_get_expected_embedding(
            dev_reference_buffer,
            output_desc,
            embedding_desc.dtype,
            dev_indices,
            indices_desc,
            wholememory::get_default_env_func(),
            stream);

          void* host_gather_buffer    = nullptr;
          void* host_reference_buffer = nullptr;
          WM_CUDA_CHECK_NO_THROW(cudaMallocHost(&host_gather_buffer, gather_buffer_size));
          WM_CUDA_CHECK_NO_THROW(cudaMallocHost(&host_reference_buffer, gather_buffer_size));

          WM_CUDA_CHECK_NO_THROW(
            cudaMemcpyAsync(host_gather_buffer,
                            dev_gather_buffer,
                            wholememory_get_memory_size_from_matrix(&output_desc),
                            cudaMemcpyDeviceToHost,
                            stream));
          WM_CUDA_CHECK_NO_THROW(
            cudaMemcpyAsync(host_reference_buffer,
                            dev_reference_buffer,
                            wholememory_get_memory_size_from_matrix(&output_desc),
                            cudaMemcpyDeviceToHost,
                            stream));
          WM_CUDA_CHECK_NO_THROW(cudaGetLastError());
          WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));

          wholememory_ops::testing::host_check_embedding_same(
            host_gather_buffer, output_desc, host_reference_buffer, output_desc);

          WM_CUDA_CHECK_NO_THROW(cudaFreeHost(host_gather_buffer));
          WM_CUDA_CHECK_NO_THROW(cudaFreeHost(host_reference_buffer));
          WM_CUDA_CHECK_NO_THROW(cudaFree(dev_reference_buffer));
        }
      }
      WHOLEMEMORY_CHECK_NOTHROW(wholememory_communicator_barrier(wm_comm) == WHOLEMEMORY_SUCCESS);

      // run many times
      auto one_gather_fun = [&](int64_t block_count, bool use_old) {
        WHOLEMEMORY_CHECK_NOTHROW(
          gather_fun_with_block_count(
            embedding_tensor, indices_tensor, output_tensor, block_count, use_old, stream) ==
          WHOLEMEMORY_SUCCESS);
        WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
      };
      auto run_loop_gather = [&](bool use_old) {
        for (int64_t block_count = params.block_min; block_count <= params.block_max;
             block_count += params.block_step) {
          if (this_rank_been_selected) {
            struct timeval tv_run_s, tv_run_e;
            gettimeofday(&tv_run_s, nullptr);

            for (int64_t iter = 0; iter < params.loop_count; iter++) {
              one_gather_fun(block_count, use_old);
            }

            gettimeofday(&tv_run_e, nullptr);
            int64_t real_time_used_us = TIME_DIFF_US(tv_run_s, tv_run_e);
            double single_run_time_us = double(real_time_used_us) / params.loop_count;
            double bw = params.get_gather_size() / (1e9) / (single_run_time_us / 1e6);
            printf(
              " rank: %d  , gather_mode= %d, old_gather = %d , embedding_table_size = %lf GB, "
              "gather_size= %lf GB, block_count=%ld "
              ", "
              "bw = %.2lf GB/s \n",
              wm_comm->world_rank,
              int(params.gather_mode),
              int(use_old),
              params.get_embedding_table_size() / (1024 * 1024 * 1024.0),
              params.get_gather_size() / (1024 * 1024 * 1024.0),
              block_count,
              bw);
          }

          WHOLEMEMORY_CHECK_NOTHROW(wholememory_communicator_barrier(wm_comm) ==
                                    WHOLEMEMORY_SUCCESS);
        }
      };
      run_loop_gather(true);
      printf("**********************************\n\n\n\n\n***********************************\n");
      run_loop_gather(false);

      if (this_rank_been_selected) {
        WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(indices_tensor) ==
                                  WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(output_tensor) == WHOLEMEMORY_SUCCESS);

        WM_CUDA_CHECK_NO_THROW(cudaFreeHost(host_indices));
        WM_CUDA_CHECK_NO_THROW(cudaFree(dev_indices));
        WM_CUDA_CHECK_NO_THROW(cudaFree(dev_gather_buffer));
      }
      //
      WHOLEMEMORY_CHECK_NOTHROW(wholememory_destroy_tensor(embedding_tensor) ==
                                WHOLEMEMORY_SUCCESS);

      WHOLEMEMORY_CHECK_NOTHROW(wholememory::destroy_all_communicators() == WHOLEMEMORY_SUCCESS);

      WHOLEMEMORY_CHECK_NOTHROW(wholememory_finalize() == WHOLEMEMORY_SUCCESS);
    },

    true);
}

}  // namespace wholegraph::bench::gather_scatter

static double parsesize(const char* value)
{
  long long int units;
  double size;
  char size_lit;

  int count = sscanf(value, "%lf %1s", &size, &size_lit);

  switch (count) {
    case 2:
      switch (size_lit) {
        case 'G':
        case 'g': units = 1024 * 1024 * 1024; break;
        case 'M':
        case 'm': units = 1024 * 1024; break;
        case 'K':
        case 'k': units = 1024; break;
        default: return -1.0;
      };
      break;
    case 1: units = 1; break;
    default: return -1.0;
  }

  return size * units;
}

int main(int argc, char** argv)
{
  wholegraph::bench::gather_scatter::SmGatherParam params;

  const char* optstr = "ht:l:e:g:d:c:r:s:n:a:p:f:k:";

  struct option opts[] = {
    {"help", no_argument, NULL, 'h'},
    {"memory_type",
     required_argument,
     NULL,
     't'},  // 0: None, 1: Continuous, 2: Chunked, 3 Distributed
    {"memory_location", required_argument, NULL, 'l'},  // 0: None, 1: Device, 2: Host
    {"embedding_table_size", required_argument, NULL, 'e'},
    {"gather_size", required_argument, NULL, 'g'},
    {"embedding_dim", required_argument, NULL, 'd'},
    {"loop_count", required_argument, NULL, 'c'},
    {"gather_mode", required_argument, NULL, 'f'},  // gather_mode: gather or scatter
    {"check_result", required_argument, NULL, 'k'},
    {"node_rank", required_argument, NULL, 'r'},    // node_rank
    {"node_size", required_argument, NULL, 's'},    // node_size
    {"num_gpu", required_argument, NULL, 'n'},      // num gpu per node
    {"server_addr", required_argument, NULL, 'a'},  // server_addr
    {"server_port", required_argument, NULL, 'p'}   // server_port
  };

  const char* usage =
    "Usage: %s [options]\n"
    "Options:\n"
    "  -h, --help      display this help and exit\n"
    "  -t, --memory_type   specify wholememory type, 0: None, 1: Continuous, 2: Chunked, 3: "
    "Distributed\n"
    "  -l, --memory_location    specify wholememory location, 0: None, 1: Device, 2: Host\n"
    "  -e, --embedding_table_size    specify embedding table size\n"
    "  -g, --gather_size    specify gather size\n"
    "  -d, --embedding_dim    specify embedding dimension\n"
    "  -c, --loop_count    specify loop count\n"
    "  -f, --test_type    specify gather mode,  0: kSingeReadSingle, 1: kSingleReadAllExcludeSelf, "
    "2: kAllExcludeSelfReadSingle 3: kAllReadAll \n"
    "  -k, --check_result  need check_result, 0: don't check result, 1:check result\n"
    "  -r, --node_rank    node_rank of current process\n"
    "  -s, --node_size    node_size or process count\n"
    "  -n, --num_gpu   num_gpu per process\n"
    "  -a, --server_addr    specify sideband server address\n"
    "  -p, --server_port    specify sideband server port\n";

  /**
   * /**
    * enum class GatherMode {
  kSingleReadSingle = 0,
  kSingleReadAllExcludeSelf,
  kAllExcludeSelfReadSingle,
  kAllReadAll
  };
   */

  int c;
  bool has_option = false;
  while ((c = getopt_long(argc, argv, optstr, opts, NULL)) != -1) {
    has_option = true;
    switch (c) {
      char* endptr;
      long val;
      case 'h': printf(usage, argv[0]); exit(EXIT_SUCCESS);
      case 't':
        val = strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val < 0 || val > 3) {
          printf("Invalid argument for option -t\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_memory_type(static_cast<wholememory_memory_type_t>(val));
        break;
      case 'l':
        val = strtol(optarg, &endptr, 10);
        if (*endptr != '\0' || val < 0 || val > 2) {
          printf("Invalid argument for option -l\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_memory_location(static_cast<wholememory_memory_location_t>(val));
        break;
      case 'e':
        val = long(parsesize(optarg));
        if (val < 0) {
          printf("Negative value, invalid argument for option -e\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_embedding_table_size(val);
        break;
      case 'g':
        val = long(parsesize(optarg));
        if (val < 0) {
          printf("Negative value, invalid argument for option -g\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_gather_size(val);
        break;
      case 'd':
        val = std::stoll(optarg);
        if (val < 0) {
          printf("Negative value, invalid argument for option -d\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_embedding_dim(val);
        break;
      case 'c':
        val = std::stoi(optarg);
        if (val < 0) {
          printf("Negative value, invalid argument for option -c\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_loop_count(val);
        break;

      case 'f':
        val = std::stoi(optarg);

        if (val < 0 || val >= int(wholegraph::bench::gather_scatter::GatherMode::kGatherModeNum)) {
          printf("Invalid argument for option -f\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_gather_mode(static_cast<wholegraph::bench::gather_scatter::GatherMode>(val));
        break;
      case 'k':
        val = std::stoi(optarg);

        if (val < 0 || val > 1) {
          printf("Invalid argument for option -f\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_need_check_result(val);
        break;
      case 'a': params.set_server_addr(optarg); break;
      case 'p':
        val = std::atoi(optarg);
        if (val < 0) {
          printf("Negative value, invalid argument for option -p\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_server_port(val);
        break;
      case 'r':
        val = std::atoi(optarg);
        if (val < 0) {
          printf("Negative value, invalid argument for option -r\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_node_rank(val);
        break;
      case 's':
        val = std::atoi(optarg);
        if (val < 0) {
          printf("Negative value, invalid argument for option -s\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_node_size(val);
        break;
      case 'n':
        val = std::atoi(optarg);
        if (val < 0) {
          printf("Negative value, invalid argument for option -n\n");
          printf(usage, argv[0]);
          exit(EXIT_FAILURE);
        }
        params.set_num_gpu(val);
        break;
      default:
        printf("Invalid or unrecognized option\n");
        printf(usage, argv[0]);
        exit(EXIT_FAILURE);
    }
  }
  if (!has_option) { printf("No option or argument is passed, use the default param\n"); }

  wholegraph::bench::gather_scatter::sm_count_gather_benchmark(params);

  printf(" block count gather finish\n");

  return 0;
}
