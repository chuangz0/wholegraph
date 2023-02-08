#include "graph_sampling_test_utils.hpp"

#include <experimental/random>
#include <gtest/gtest.h>
#include <iterator>
#include <random>

#include "wholememory_ops/raft_random.cuh"
#include <wholememory_ops/register.hpp>

namespace wholegraph_ops::testing {

template <typename DataType>
void host_get_csr_graph(int64_t graph_node_count,
                        int64_t graph_edge_count,
                        void* host_csr_row_ptr,
                        wholememory_array_description_t graph_csr_row_ptr_desc,
                        void* host_csr_col_ptr,
                        wholememory_array_description_t graph_csr_col_ptr_desc)
{
  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  DataType* csr_col_ptr         = static_cast<DataType*>(host_csr_col_ptr);
  int64_t average_edge_per_node = graph_edge_count / graph_node_count;

  std::default_random_engine generator;
  std::binomial_distribution<int64_t> distribution(average_edge_per_node, 1);

  int total_edge = 0;

  for (int64_t i = 0; i < graph_node_count; i++) {
    while (true) {
      int64_t random_num = distribution(generator);
      if (random_num >= 0 && random_num <= graph_node_count) {
        csr_row_ptr[i] = random_num;
        total_edge += random_num;
        break;
      }
    }
  }

  int64_t adjust_edge = std::abs(total_edge - graph_edge_count);
  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_int_distribution<int64_t> distr(0, graph_node_count - 1);
  if (total_edge > graph_edge_count) {
    for (int64_t i = 0; i < adjust_edge; i++) {
      while (true) {
        int64_t random_row_id = distr(gen);
        if (csr_row_ptr[random_row_id] > 0) {
          csr_row_ptr[random_row_id]--;
          break;
        }
      }
    }
  }
  if (total_edge < graph_edge_count) {
    for (int64_t i = 0; i < adjust_edge; i++) {
      while (true) {
        int64_t random_row_id = distr(gen);
        if (csr_row_ptr[random_row_id] < graph_node_count) {
          csr_row_ptr[random_row_id]++;
          break;
        }
      }
    }
  }

  host_prefix_sum_array(host_csr_row_ptr, graph_csr_row_ptr_desc);

  EXPECT_TRUE(csr_row_ptr[graph_node_count] == graph_edge_count);

  for (int64_t i = 0; i < graph_node_count; i++) {
    int64_t start      = csr_row_ptr[i];
    int64_t end        = csr_row_ptr[i + 1];
    int64_t edge_count = end - start;
    if (edge_count == 0) continue;
    std::vector<int64_t> array_out(edge_count);
    std::vector<int64_t> array_in(graph_node_count);
    for (int64_t i = 0; i < graph_node_count; i++) {
      array_in[i] = i;
    }

    std::sample(array_in.begin(), array_in.end(), array_out.begin(), edge_count, gen);
    for (int j = 0; j < edge_count; j++) {
      csr_col_ptr[start + j] = (DataType)array_out[j];
    }
  }
}

void gen_csr_graph(int64_t graph_node_count,
                   int64_t graph_edge_count,
                   void* host_csr_row_ptr,
                   wholememory_array_description_t graph_csr_row_ptr_desc,
                   void* host_csr_col_ptr,
                   wholememory_array_description_t graph_csr_col_ptr_desc)
{
  EXPECT_TRUE(graph_csr_row_ptr_desc.dtype == WHOLEMEMORY_DT_INT64);

  if (graph_csr_col_ptr_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_csr_graph<int64_t>(graph_node_count,
                                graph_edge_count,
                                host_csr_row_ptr,
                                graph_csr_row_ptr_desc,
                                host_csr_col_ptr,
                                graph_csr_col_ptr_desc);

  } else if (graph_csr_col_ptr_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_csr_graph<int>(graph_node_count,
                            graph_edge_count,
                            host_csr_row_ptr,
                            graph_csr_row_ptr_desc,
                            host_csr_col_ptr,
                            graph_csr_col_ptr_desc);
  }
}

template <typename DataType>
void host_get_random_array(void* array,
                           wholememory_array_description_t array_desc,
                           int64_t low,
                           int64_t high)
{
  DataType* array_ptr = static_cast<DataType*>(array);
  std::experimental::reseed();
  for (int64_t i = 0; i < array_desc.size; i++) {
    DataType random_num                      = std::experimental::randint<DataType>(low, high);
    array_ptr[i + array_desc.storage_offset] = random_num;
  }
}

void host_random_init_array(void* array,
                            wholememory_array_description_t array_desc,
                            int64_t low,
                            int64_t high)
{
  EXPECT_TRUE(array_desc.dtype == WHOLEMEMORY_DT_INT || array_desc.dtype == WHOLEMEMORY_DT_INT64);
  if (array_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_random_array<int>(array, array_desc, low, high);
  } else {
    host_get_random_array<int64_t>(array, array_desc, low, high);
  }
}

template <typename DataType>
void host_get_prefix_sum_array(void* array, wholememory_array_description_t array_desc)
{
  DataType* array_ptr = static_cast<DataType*>(array);
  if (array_desc.size <= 0) return;
  DataType old_value = array_ptr[0];
  array_ptr[0]       = 0;
  for (int64_t i = 1; i < array_desc.size; i++) {
    DataType tmp = array_ptr[i];
    array_ptr[i] = array_ptr[i - 1] + old_value;
    old_value    = tmp;
  }
}

void host_prefix_sum_array(void* array, wholememory_array_description_t array_desc)
{
  EXPECT_TRUE(array_desc.dtype == WHOLEMEMORY_DT_INT || array_desc.dtype == WHOLEMEMORY_DT_INT64);
  if (array_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_prefix_sum_array<int>(array, array_desc);
  } else {
    host_get_prefix_sum_array<int64_t>(array, array_desc);
  }
}

void copy_host_array_to_wholememory(void* host_array,
                                    wholememory_handle_t array_handle,
                                    wholememory_array_description_t array_desc,
                                    cudaStream_t stream)
{
  void* local_array_ptr;
  size_t local_array_size, local_array_offset;
  EXPECT_EQ(wholememory_get_local_memory(
              &local_array_ptr, &local_array_size, &local_array_offset, array_handle),
            WHOLEMEMORY_SUCCESS);
  int64_t array_ele_size = wholememory_dtype_get_element_size(array_desc.dtype);
  EXPECT_EQ(local_array_size % array_ele_size, 0);
  EXPECT_EQ(local_array_offset % array_ele_size, 0);
  wholememory_comm_t wm_comm;
  EXPECT_EQ(wholememory_get_communicator(&wm_comm, array_handle), WHOLEMEMORY_SUCCESS);

  if (local_array_size) {
    EXPECT_EQ(cudaMemcpyAsync(local_array_ptr,
                              static_cast<char*>(host_array) + local_array_offset,
                              local_array_size,
                              cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
  wholememory_communicator_barrier(wm_comm);
}

template <typename DataType>
void host_get_sample_offset(void* host_csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_desc,
                            void* host_center_nodes,
                            wholememory_array_description_t center_node_desc,
                            int max_sample_count,
                            void* host_ref_output_sample_offset,
                            wholememory_array_description_t output_sample_offset_desc)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  DataType* center_nodes_ptr    = static_cast<DataType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  for (int64_t i = 0; i < center_node_desc.size; i++) {
    DataType center_node_id = center_nodes_ptr[i];
    int neighbor_node_count = csr_row_ptr[center_node_id + 1] - csr_row_ptr[center_node_id];
    if (max_sample_count > 0) {
      neighbor_node_count = std::min(neighbor_node_count, max_sample_count);
    }
    output_sample_offset_ptr[i] = neighbor_node_count;
  }
}

template <typename IdType, typename ColIdType>
void host_sample_all(void* host_csr_row_ptr,
                     wholememory_array_description_t csr_row_ptr_desc,
                     void* host_csr_col_ptr,
                     wholememory_array_description_t csr_col_ptr_desc,
                     void* host_center_nodes,
                     wholememory_array_description_t center_node_desc,
                     int max_sample_count,
                     void* host_ref_output_sample_offset,
                     wholememory_array_description_t output_sample_offset_desc,
                     void* host_ref_output_dest_nodes,
                     void* host_ref_output_center_nodes_local_id,
                     void* host_ref_output_global_edge_id)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  ColIdType* csr_col_ptr        = static_cast<ColIdType*>(host_csr_col_ptr);
  IdType* center_nodes_ptr      = static_cast<IdType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  ColIdType* output_dest_nodes_ptr      = static_cast<ColIdType*>(host_ref_output_dest_nodes);
  int* output_center_nodes_local_id_ptr = static_cast<int*>(host_ref_output_center_nodes_local_id);
  int64_t* output_global_edge_id_ptr    = static_cast<int64_t*>(host_ref_output_global_edge_id);

  int64_t center_nodes_count = center_node_desc.size;

  for (int64_t i = 0; i < center_nodes_count; i++) {
    int output_id         = output_sample_offset_ptr[i];
    int output_local_id   = 0;
    IdType center_node_id = center_nodes_ptr[i];
    for (int64_t j = csr_row_ptr[center_node_id]; j < csr_row_ptr[center_node_id + 1]; j++) {
      output_dest_nodes_ptr[output_id + output_local_id]            = csr_col_ptr[j];
      output_center_nodes_local_id_ptr[output_id + output_local_id] = (int)i;
      output_global_edge_id_ptr[output_id + output_local_id]        = j;
      output_local_id++;
    }
  }
}

REGISTER_DISPATCH_TWO_TYPES(HOSTSAMPLEALL, host_sample_all, SINT3264, SINT3264)

template <int Offset = 0>
void random_sample_without_replacement_cpu_base(std::vector<int>* a,
                                                const std::vector<uint32_t>& r,
                                                int M,
                                                int N)
{
  a->resize(M + Offset);
  std::vector<int> Q(N + Offset);
  for (int i = Offset; i < N + Offset; ++i) {
    Q[i] = i;
  }
  for (int i = Offset; i < M + Offset; ++i) {
    a->at(i) = Q[r[i]];
    Q[r[i]]  = Q[N - i + 2 * Offset - 1];
  }
}

void random_sample_without_replacement_cpu_base_2(std::vector<int>& a,
                                                  const std::vector<int>& r,
                                                  int M,
                                                  int N)
{
  std::vector<int> Q(N);
  for (int i = 0; i < N; ++i) {
    Q[i] = i;
  }
  for (int i = 0; i < M; ++i) {
    a[i]    = Q[r[i]];
    Q[r[i]] = Q[N - i - 1];
  }
}

template <typename IdType, typename ColIdType>
void host_unweighted_sample_without_replacement(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_center_nodes,
  wholememory_array_description_t center_node_desc,
  int max_sample_count,
  void* host_ref_output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* host_ref_output_dest_nodes,
  void* host_ref_output_center_nodes_local_id,
  void* host_ref_output_global_edge_id,
  unsigned long long random_seed)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);

  int64_t* csr_row_ptr          = static_cast<int64_t*>(host_csr_row_ptr);
  ColIdType* csr_col_ptr        = static_cast<ColIdType*>(host_csr_col_ptr);
  IdType* center_nodes_ptr      = static_cast<IdType*>(host_center_nodes);
  int* output_sample_offset_ptr = static_cast<int*>(host_ref_output_sample_offset);

  ColIdType* output_dest_nodes_ptr      = static_cast<ColIdType*>(host_ref_output_dest_nodes);
  int* output_center_nodes_local_id_ptr = static_cast<int*>(host_ref_output_center_nodes_local_id);
  int64_t* output_global_edge_id_ptr    = static_cast<int64_t*>(host_ref_output_global_edge_id);

  int64_t center_nodes_count = center_node_desc.size;

  int M = max_sample_count;

  static const int warp_count_array[32]       = {1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8,
                                                 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
  static const int items_per_thread_array[32] = {1, 2, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  int func_idx                                = (max_sample_count - 1) / 32;
  int device_num_threads                      = warp_count_array[func_idx] * 32;
  int items_per_thread                        = items_per_thread_array[func_idx];

  for (int64_t i = 0; i < center_nodes_count; i++) {
    int output_id          = output_sample_offset_ptr[i];
    int output_local_id    = 0;
    IdType center_node_id  = center_nodes_ptr[i];
    int64_t start          = csr_row_ptr[center_node_id];
    int64_t end            = csr_row_ptr[center_node_id + 1];
    int64_t neighbor_count = end - start;
    int N                  = neighbor_count;
    int blockidx           = i;
    int gidx               = blockidx * device_num_threads;

    if (neighbor_count <= 0) continue;

    if (neighbor_count <= max_sample_count) {
      for (int64_t j = start; j < end; j++) {
        output_dest_nodes_ptr[output_id + output_local_id]            = csr_col_ptr[j];
        output_center_nodes_local_id_ptr[output_id + output_local_id] = (int)i;
        output_global_edge_id_ptr[output_id + output_local_id]        = j;
        output_local_id++;
      }
    } else {
      std::vector<uint32_t> r(neighbor_count);
      for (int j = 0; j < device_num_threads; j++) {
        int local_gidx = gidx + j;
        PCGenerator rng(random_seed, (uint64_t)local_gidx, (uint64_t)0);

        for (int k = 0; k < items_per_thread; k++) {
          int id = k * device_num_threads + j;
          uint32_t random_num;
          rng.next(random_num);
          if (id < neighbor_count) { r[id] = id < M ? (random_num % (N - id)) : N; }
        }
      }

      std::vector<int> random_sample_id(max_sample_count, 0);
      random_sample_without_replacement_cpu_base(&random_sample_id, r, M, N);
      for (int sample_id = 0; sample_id < M; sample_id++) {
        output_dest_nodes_ptr[output_id + sample_id] =
          csr_col_ptr[start + random_sample_id[sample_id]];
        output_center_nodes_local_id_ptr[output_id + sample_id] = i;
        output_global_edge_id_ptr[output_id + sample_id] = start + random_sample_id[sample_id];
      }
    }
  }
}

REGISTER_DISPATCH_TWO_TYPES(HOSTUNWEIGHTEDSAMPLEWITHOUTREPLACEMENT,
                            host_unweighted_sample_without_replacement,
                            SINT3264,
                            SINT3264)

void wholegraph_csr_unweighted_sample_without_replacement_cpu(
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_center_nodes,
  wholememory_array_description_t center_node_desc,
  int max_sample_count,
  void** host_ref_output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void** host_ref_output_dest_nodes,
  void** host_ref_output_center_nodes_local_id,
  void** host_ref_output_global_edge_id,
  int* output_sample_dest_nodes_count,
  unsigned long long random_seed)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT64);
  EXPECT_EQ(output_sample_offset_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_sample_offset_desc.size, center_node_desc.size + 1);

  *host_ref_output_sample_offset =
    (void*)malloc(wholememory_get_memory_size_from_array(&output_sample_offset_desc));

  if (center_node_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_sample_offset<int64_t>(host_csr_row_ptr,
                                    csr_row_ptr_desc,
                                    host_center_nodes,
                                    center_node_desc,
                                    max_sample_count,
                                    *host_ref_output_sample_offset,
                                    output_sample_offset_desc);
  } else if (center_node_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_sample_offset<int>(host_csr_row_ptr,
                                csr_row_ptr_desc,
                                host_center_nodes,
                                center_node_desc,
                                max_sample_count,
                                *host_ref_output_sample_offset,
                                output_sample_offset_desc);
  }
  host_prefix_sum_array(*host_ref_output_sample_offset, output_sample_offset_desc);
  *output_sample_dest_nodes_count =
    static_cast<int*>(*host_ref_output_sample_offset)[center_node_desc.size];

  *host_ref_output_dest_nodes            = malloc((*output_sample_dest_nodes_count) *
                                       wholememory_dtype_get_element_size(csr_col_ptr_desc.dtype));
  *host_ref_output_center_nodes_local_id = malloc((*output_sample_dest_nodes_count) * sizeof(int));
  *host_ref_output_global_edge_id = malloc((*output_sample_dest_nodes_count) * sizeof(int64_t));

  if (max_sample_count <= 0) {
    DISPATCH_TWO_TYPES(center_node_desc.dtype,
                       csr_col_ptr_desc.dtype,
                       HOSTSAMPLEALL,
                       host_csr_row_ptr,
                       csr_row_ptr_desc,
                       host_csr_col_ptr,
                       csr_col_ptr_desc,
                       host_center_nodes,
                       center_node_desc,
                       max_sample_count,
                       *host_ref_output_sample_offset,
                       output_sample_offset_desc,
                       *host_ref_output_dest_nodes,
                       *host_ref_output_center_nodes_local_id,
                       *host_ref_output_global_edge_id);
    return;
  }
  if (max_sample_count > 1024) { return; }

  DISPATCH_TWO_TYPES(center_node_desc.dtype,
                     csr_col_ptr_desc.dtype,
                     HOSTUNWEIGHTEDSAMPLEWITHOUTREPLACEMENT,
                     host_csr_row_ptr,
                     csr_row_ptr_desc,
                     host_csr_col_ptr,
                     csr_col_ptr_desc,
                     host_center_nodes,
                     center_node_desc,
                     max_sample_count,
                     *host_ref_output_sample_offset,
                     output_sample_offset_desc,
                     *host_ref_output_dest_nodes,
                     *host_ref_output_center_nodes_local_id,
                     *host_ref_output_global_edge_id,
                     random_seed);
}

template <typename DataType>
void check_value_same(void* value, void* ref, int64_t size)
{
  int64_t diff_count;

  DataType* value_ptr = static_cast<DataType*>(value);
  DataType* ref_ptr   = static_cast<DataType*>(ref);

  for (int i = 0; i < size; i++) {
    if (value_ptr[i] != ref_ptr[i]) {
      if (diff_count < 10) {
        printf("value = %ld, ref = %ld\n",
               static_cast<int64_t>(value_ptr[i]),
               static_cast<int64_t>(ref_ptr[i]));
        EXPECT_EQ(value_ptr[i], ref_ptr[i]);
      }
      diff_count++;
    }
  }
}

REGISTER_DISPATCH_ONE_TYPE(CHECKVALUESAME, check_value_same, SINT3264)

void host_check_two_array_same(void* host_array,
                               wholememory_array_description_t host_array_desc,
                               void* host_ref,
                               wholememory_array_description_t host_ref_desc)
{
  EXPECT_EQ(host_array_desc.dtype, host_ref_desc.dtype);
  EXPECT_EQ(host_array_desc.size, host_ref_desc.size);
  DISPATCH_ONE_TYPE(
    host_array_desc.dtype, CHECKVALUESAME, host_array, host_ref, host_array_desc.size);
}

}  // namespace wholegraph_ops::testing