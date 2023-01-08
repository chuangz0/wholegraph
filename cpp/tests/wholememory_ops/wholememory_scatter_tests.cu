#include <gtest/gtest.h>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_op.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/env_func_ptrs.hpp"

#include "../wholememory/wholememory_test_utils.hpp"
#include "embedding_test_utils.hpp"

typedef struct WholeMemoryScatterTestParam {
  wholememory_matrix_description_t get_embedding_desc() const {
    int64_t matrix_sizes[2] = {embedding_dim, embedding_entry_count};
    return wholememory_create_matrix_desc(matrix_sizes, embedding_stride, embedding_storage_offset, embedding_type);
  }
  wholememory_array_description_t get_indices_desc() const {
    return wholememory_create_array_desc(indices_count, indices_storage_offset, indices_type);
  }
  wholememory_matrix_description_t get_input_desc() const {
    int64_t input_sizes[2] = {embedding_dim, indices_count};
    return wholememory_create_matrix_desc(input_sizes, input_stride, input_storage_offset, input_type);
  }
  int64_t get_embedding_granularity() const {
    return embedding_stride * wholememory_dtype_get_element_size(embedding_type);
  }

  WholeMemoryScatterTestParam& set_memory_type(wholememory_memory_type_t new_memory_type) {
    memory_type = new_memory_type;
    return *this;
  }
  WholeMemoryScatterTestParam& set_memory_location(wholememory_memory_location_t new_memory_location) {
    memory_location = new_memory_location;
    return *this;
  }
  WholeMemoryScatterTestParam& set_entry_count(int64_t entry_count) {
    embedding_entry_count = entry_count;
    return *this;
  }
  WholeMemoryScatterTestParam& set_embedding_dim(int64_t new_embedding_dim) {
    embedding_dim = new_embedding_dim;
    if (embedding_stride < embedding_dim) embedding_stride = embedding_dim;
    if (input_stride < embedding_dim) input_stride = embedding_dim;
    return *this;
  }
  WholeMemoryScatterTestParam& set_embedding_stride(int64_t new_embedding_stride) {
    embedding_stride = new_embedding_stride;
    return *this;
  }
  WholeMemoryScatterTestParam& set_input_stride(int64_t new_input_stride) {
    input_stride = new_input_stride;
    return *this;
  }
  WholeMemoryScatterTestParam& set_embedding_type(wholememory_dtype_t new_embedding_type) {
    embedding_type = new_embedding_type;
    return *this;
  }
  WholeMemoryScatterTestParam& set_indices_type(wholememory_dtype_t new_indices_type) {
    indices_type = new_indices_type;
    return *this;
  }
  WholeMemoryScatterTestParam& set_input_type(wholememory_dtype_t new_input_type) {
    input_type = new_input_type;
    return *this;
  }
  wholememory_memory_type_t memory_type = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location = WHOLEMEMORY_ML_DEVICE;
  int64_t embedding_entry_count = 1000000LL;
  int64_t embedding_dim = 32;
  int64_t embedding_stride = 32;
  int64_t indices_count = 100000;
  int64_t input_stride = 32;
  wholememory_dtype_t embedding_type = WHOLEMEMORY_DT_FLOAT;
  wholememory_dtype_t indices_type = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t input_type = WHOLEMEMORY_DT_FLOAT;
  int64_t embedding_storage_offset = 0;
  int64_t indices_storage_offset = 0;
  int64_t input_storage_offset = 0;
} WholeMemoryScatterTestParam;

class WholeMemoryScatterParameterTests : public ::testing::TestWithParam<WholeMemoryScatterTestParam> {
};

TEST_P(WholeMemoryScatterParameterTests, ScatterTest) {
  auto params = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(dev_count, [&params, &pipes](int world_rank, int world_size) {
    EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);

    wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, world_rank, world_size);

    wholememory_handle_t embedding_handle;
    auto embedding_desc = params.get_embedding_desc();
    auto indices_desc = params.get_indices_desc();
    auto input_desc = params.get_input_desc();
    size_t embedding_entry_size = params.get_embedding_granularity();
    EXPECT_EQ(wholememory_malloc(&embedding_handle,
                                 wholememory_get_memory_size_from_matrix(&embedding_desc),
                                 wm_comm,
                                 params.memory_type,
                                 params.memory_location,
                                 embedding_entry_size), WHOLEMEMORY_SUCCESS);

    cudaStream_t stream;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    void  *host_indices, *dev_indices, *dev_input_buffer, *dev_gather_buffer, *host_gather_buffer, *host_input_buffer;
    size_t scatter_buffer_size = wholememory_get_memory_size_from_matrix(&input_desc);
    size_t indices_buffer_size = wholememory_get_memory_size_from_array(&indices_desc);

    EXPECT_EQ(cudaMallocHost(&host_indices, indices_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_indices, indices_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_input_buffer, scatter_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dev_gather_buffer, scatter_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMallocHost(&host_input_buffer, scatter_buffer_size), cudaSuccess);
    EXPECT_EQ(cudaMallocHost(&host_gather_buffer, scatter_buffer_size), cudaSuccess);

    wholememory_ops::testing::host_random_init_indices(host_indices, indices_desc);
    EXPECT_EQ(cudaMemcpyAsync(dev_indices,
                              host_indices,
                              wholememory_get_memory_size_from_array(&indices_desc),
                              cudaMemcpyHostToDevice,
                              stream), cudaSuccess);
    wholememory_ops::testing::device_get_expected_embedding(dev_input_buffer,
                                                            input_desc,
                                                            embedding_desc.dtype,
                                                            dev_indices,
                                                            indices_desc,
                                                            wholememory::get_default_env_func(),
                                                            stream);

    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    wholememory_communicator_barrier(wm_comm);

    EXPECT_EQ(wholememory_scatter(dev_input_buffer, input_desc,
                                  dev_indices, indices_desc,
                                  embedding_handle, embedding_desc,
                                  wholememory::get_default_env_func(), stream), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    wholememory_communicator_barrier(wm_comm);

    EXPECT_EQ(wholememory_gather(embedding_handle, embedding_desc,
                                 dev_indices, indices_desc,
                                 dev_gather_buffer, input_desc,
                                 wholememory::get_default_env_func(), stream), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(cudaMemcpyAsync(host_gather_buffer,
                              dev_gather_buffer,
                              wholememory_get_memory_size_from_matrix(&input_desc),
                              cudaMemcpyDeviceToHost,
                              stream), cudaSuccess);
    EXPECT_EQ(cudaMemcpyAsync(host_input_buffer,
                              dev_input_buffer,
                              wholememory_get_memory_size_from_matrix(&input_desc),
                              cudaMemcpyDeviceToHost,
                              stream), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    wholememory_ops::testing::host_check_embedding_same(host_gather_buffer,
                                                        input_desc,
                                                        host_input_buffer,
                                                        input_desc);

    EXPECT_EQ(cudaFreeHost(host_indices), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_indices), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_input_buffer), cudaSuccess);
    EXPECT_EQ(cudaFree(dev_gather_buffer), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host_input_buffer), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host_gather_buffer), cudaSuccess);

    EXPECT_EQ(wholememory_free(embedding_handle), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
  });
}

INSTANTIATE_TEST_CASE_P(
    WholeMemoryScatterOpTests,
    WholeMemoryScatterParameterTests,
    ::testing::Values(
#if 1
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_memory_location(WHOLEMEMORY_ML_HOST),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_memory_location(WHOLEMEMORY_ML_HOST),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_memory_location(WHOLEMEMORY_ML_HOST),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(128),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(128),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(128),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(127),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(127),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(127),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(129),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(129),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(129),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(513),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(513),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(513),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_input_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_input_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_input_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_type(WHOLEMEMORY_DT_HALF).set_input_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_type(WHOLEMEMORY_DT_HALF).set_input_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_type(WHOLEMEMORY_DT_HALF).set_input_type(WHOLEMEMORY_DT_HALF),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_indices_type(WHOLEMEMORY_DT_INT64),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_indices_type(WHOLEMEMORY_DT_INT64),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_indices_type(WHOLEMEMORY_DT_INT64),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_input_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_input_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_input_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_type(WHOLEMEMORY_DT_HALF).set_embedding_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_type(WHOLEMEMORY_DT_HALF).set_embedding_stride(33),
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_type(WHOLEMEMORY_DT_HALF).set_embedding_stride(33),
#endif
        WholeMemoryScatterTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
    ));

