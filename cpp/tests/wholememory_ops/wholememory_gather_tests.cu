#include <gtest/gtest.h>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/initialize.hpp"

#include "../wholememory/wholememory_test_utils.hpp"

typedef struct SingleGPUGatherTestParam {
  wholememory_matrix_description_t get_embedding_desc() const {
    int64_t matrix_sizes[2] = {embedding_dim, embedding_entry_count};
    return wholememory_create_matrix_desc(matrix_sizes, embedding_stride, embedding_storage_offset, embedding_type);
  }
  wholememory_array_description_t get_indice_desc() const {
    return wholememory_create_array_desc(indice_count, indice_storage_offset, indice_type);
  }
  wholememory_matrix_description_t get_output_desc() const {
    int64_t output_sizes[2] = {embedding_dim, indice_count};
    return wholememory_create_matrix_desc(output_sizes, output_stride, output_storage_offset, output_type);
  }
  int64_t get_embedding_granularity() const {
    return embedding_stride * wholememory_dtype_get_element_size(embedding_type);
  }
  wholememory_memory_type_t memory_type = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location = WHOLEMEMORY_ML_DEVICE;
  int64_t embedding_entry_count = 10000000LL;
  int64_t embedding_dim = 128;
  int64_t embedding_stride = 128;
  int64_t indice_count = 1000000;
  int64_t output_stride = 128;
  int64_t embedding_storage_offset = 0;
  int64_t indice_storage_offset = 0;
  int64_t output_storage_offset = 0;
  wholememory_dtype_t embedding_type = WHOLEMEMORY_DT_FLOAT;
  wholememory_dtype_t indice_type = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t output_type = WHOLEMEMORY_DT_FLOAT;
} SingleGPUGatherTestParam;

class WholeMemorySingleGPUGatherParameterTests : public ::testing::TestWithParam<SingleGPUGatherTestParam> {
};

TEST_P(WholeMemorySingleGPUGatherParameterTests, GatherTest) {
  auto params = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

  wholememory_unique_id_t unique_id;
  EXPECT_EQ(wholememory_create_unique_id(&unique_id), WHOLEMEMORY_SUCCESS);

  wholememory_comm_t wm_comm;
  EXPECT_EQ(wholememory_create_communicator(&wm_comm, unique_id, 0, 1), WHOLEMEMORY_SUCCESS);

  wholememory_handle_t embedding_handle;
  auto embedding_desc = params.get_embedding_desc();
  size_t embedding_entry_size = params.get_embedding_granularity();
  EXPECT_EQ(wholememory_malloc(&embedding_handle,
                               wholememory_get_memory_size_from_matrix(&embedding_desc),
                               wm_comm,
                               params.memory_type,
                               params.memory_location,
                               embedding_entry_size), WHOLEMEMORY_SUCCESS);

  void* local_embedding_ptr;
  size_t local_embedding_size, local_embedding_offset;

  EXPECT_EQ(wholememory_get_local_memory(&local_embedding_ptr,
                                         &local_embedding_size,
                                         &local_embedding_offset,
                                         embedding_handle), WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(local_embedding_size % embedding_entry_size, 0);
  EXPECT_EQ(local_embedding_offset % embedding_entry_size, 0);

  int64_t local_entry_start = local_embedding_offset / embedding_entry_size;
  int64_t local_entry_count = local_embedding_size / embedding_entry_size;

  cudaStream_t stream;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);



  EXPECT_EQ(wholememory_free(embedding_handle), WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

  EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
}


