#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <raft/util/integer_utils.hpp>
#include <wholememory/device_reference.cuh>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

#include "cuda_macros.hpp"
#include "error.hpp"

namespace wholememory_ops {

template <typename DataTypeT>
__device__ __forceinline__ void mov_typed_data(DataTypeT* to, const DataTypeT* from)
{
  *to = *from;
}
template <int DATA_SIZE>
__device__ __forceinline__ void mov_data(void* to, const void* from)
{
  char* ptr_to         = static_cast<char*>(to);
  const char* ptr_from = static_cast<const char*>(from);
  for (int i = 0; i < DATA_SIZE; i++) {
    ptr_to[i] = ptr_from[i];
  }
}
template <>
__device__ __forceinline__ void mov_data<1>(void* to, const void* from)
{
  mov_typed_data(static_cast<int8_t*>(to), static_cast<const int8_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<2>(void* to, const void* from)
{
  mov_typed_data(static_cast<int16_t*>(to), static_cast<const int16_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<4>(void* to, const void* from)
{
  mov_typed_data(static_cast<int32_t*>(to), static_cast<const int32_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<8>(void* to, const void* from)
{
  mov_typed_data(static_cast<int64_t*>(to), static_cast<const int64_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<16>(void* to, const void* from)
{
  mov_typed_data(static_cast<int4*>(to), static_cast<const int4*>(from));
}

template <typename DataTypeT>
class type_caster {
 public:
  using LoadTypeT  = DataTypeT;
  using StoreTypeT = DataTypeT;
  static __device__ __forceinline__ LoadTypeT convert_load_data(DataTypeT data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __forceinline__ DataTypeT convert_store_data(StoreTypeT data)
  {
    return static_cast<DataTypeT>(data);
  }
};
template <>
class type_caster<__half> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ __forceinline__ LoadTypeT convert_load_data(__half data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __forceinline__ __half convert_store_data(StoreTypeT data)
  {
    return static_cast<__half>(data);
  }
};
template <>
class type_caster<__nv_bfloat16> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ LoadTypeT convert_load_data(__nv_bfloat16 data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __nv_bfloat16 convert_store_data(StoreTypeT data)
  {
    return static_cast<__nv_bfloat16>(data);
  }
};

template <typename FromT, typename ToT>
__device__ __forceinline__ ToT convert_type(FromT from)
{
  return type_caster<ToT>::convert_store_data(type_caster<FromT>::convert_load_data(from));
}

/**
 * Determine alignment of a WholeMemory matrix, in element count, maximum 16 / element_size.
 * @param embedding_desc : wholememory_matrix_description_t matrix description.
 * @return : Alignment that can be used, in element count.
 */
inline int determine_wholememory_alignment_elt_count(
  wholememory_matrix_description_t embedding_desc)
{
  int elt_size = static_cast<int>(wholememory_dtype_get_element_size(embedding_desc.dtype));
  WHOLEMEMORY_CHECK(elt_size != -1);
  int alignment = 16 / elt_size;
  for (; alignment > 1; alignment /= 2) {
    if (embedding_desc.storage_offset % alignment == 0 &&
        embedding_desc.sizes[0] % alignment == 0 && embedding_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

/**
 * Determine alignment of normal memory, in element count, maximum 16 / element_size.
 * @param ptr : pointer to the memory.
 * @param memory_desc : wholememory_matrix_description_t matrix description.
 * @return : Alignment that can be used, in element count.
 */
inline int determine_memory_alignment_elt_count(const void* ptr,
                                                wholememory_matrix_description_t memory_desc)
{
  int elt_size = static_cast<int>(wholememory_dtype_get_element_size(memory_desc.dtype));
  WHOLEMEMORY_CHECK(elt_size != -1);
  int alignment   = 16 / elt_size;
  int64_t int_ptr = reinterpret_cast<int64_t>(ptr);
  WHOLEMEMORY_CHECK(int_ptr % elt_size == 0);
  int_ptr /= elt_size;
  int_ptr += memory_desc.storage_offset;
  for (; alignment > 1; alignment /= 2) {
    if (int_ptr % alignment == 0 && memory_desc.sizes[0] % alignment == 0 &&
        memory_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

template <typename EmbeddingT, typename IndexT, typename OutputT, int ALIGNMENT = 1>
__global__ void gather_func_kernel(wholememory_gref_t embedding_gref,
                                   wholememory_matrix_description_t embedding_desc,
                                   const IndexT* indices,
                                   int64_t indice_count,
                                   OutputT* output,
                                   wholememory_matrix_description_t output_desc)
{
  int64_t output_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  for (; output_idx < indice_count; output_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    IndexT embedding_table_idx = indices[output_idx];
    wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);
    int thread_idx           = threadIdx.x;
    int embedding_size       = embedding_desc.sizes[0];
    int64_t embedding_stride = embedding_desc.stride;
    int64_t output_stride    = output_desc.stride;
    EmbeddingT embeddings[ALIGNMENT];
    OutputT outputs[ALIGNMENT];
    OutputT* output_ptr = output + output_desc.storage_offset + output_stride * output_idx;
    int64_t embedding_offset =
      embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
    for (int emb_idx = thread_idx * ALIGNMENT; emb_idx < embedding_size;
         emb_idx += ALIGNMENT * blockDim.x) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(embeddings,
                                               &embedding_dev_ref[embedding_offset + emb_idx]);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        outputs[sub_idx] = convert_type<EmbeddingT, OutputT>(embeddings[sub_idx]);
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(output_ptr + emb_idx, outputs);
    }
  }
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void gather_temp_func(wholememory_gref_t embedding_gref,
                      wholememory_matrix_description_t embedding_desc,
                      void* indices,
                      int64_t indice_count,
                      void* output,
                      wholememory_matrix_description_t output_desc,
                      cudaStream_t stream)
{
  WHOLEMEMORY_EXPECTS(output_desc.sizes[1] == indice_count,
                      "gather_func, output shape[1]=%ld, but indice_count=%ld",
                      output_desc.sizes[1],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[0] == 0) return;
  int wm_alignment   = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment   = determine_memory_alignment_elt_count(output, output_desc);
  int alignment      = std::min<int>(wm_alignment, mm_alignment);
  int embedding_size = embedding_desc.sizes[0];
  int thread_x       = raft::div_rounding_up_safe<int>(embedding_size, alignment);
  thread_x           = std::min(thread_x, 256);
  int thread_y       = 1;
  if (thread_x < 64) {
    int power2_thread_x = 1;
    for (; power2_thread_x < thread_x; power2_thread_x *= 2)
      ;
    thread_x = power2_thread_x;
    thread_y = 64 / thread_x;
  }
  int64_t block_count_64 = indice_count / thread_y;
  int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
  dim3 block_dim(thread_x, thread_y, 1);
  void (*kernel_fn)(wholememory_gref_t,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    OutputT*,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 16>;
      break;
    }
    case 8: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 8>;
      break;
    }
    case 4: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 4>;
      break;
    }
    case 2: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 2>;
      break;
    }
    case 1: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }
  kernel_fn<<<block_count, block_dim, 0, stream>>>(embedding_gref,
                                                   embedding_desc,
                                                   static_cast<const IndexT*>(indices),
                                                   indice_count,
                                                   static_cast<OutputT*>(output),
                                                   output_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename IndexT, typename EmbeddingT, int ALIGNMENT = 1>
__global__ void scatter_func_kernel(const InputT* input,
                                    wholememory_matrix_description_t input_desc,
                                    const IndexT* indices,
                                    int64_t indice_count,
                                    wholememory_gref_t embedding_gref,
                                    wholememory_matrix_description_t embedding_desc)
{
  int64_t input_idx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  for (; input_idx < indice_count; input_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    IndexT embedding_table_idx = indices[input_idx];
    wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);
    int thread_idx           = threadIdx.x;
    int embedding_size       = embedding_desc.sizes[0];
    int64_t embedding_stride = embedding_desc.stride;
    int64_t input_stride     = input_desc.stride;
    EmbeddingT embeddings[ALIGNMENT];
    InputT inputs[ALIGNMENT];
    const InputT* input_ptr = input + input_desc.storage_offset + input_stride * input_idx;
    int64_t embedding_offset =
      embedding_desc.storage_offset + embedding_table_idx * embedding_stride;
    for (int emb_idx = thread_idx * ALIGNMENT; emb_idx < embedding_size;
         emb_idx += ALIGNMENT * blockDim.x) {
      mov_data<sizeof(InputT) * ALIGNMENT>(inputs, input_ptr + emb_idx);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        embeddings[sub_idx] = convert_type<InputT, EmbeddingT>(inputs[sub_idx]);
      }
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embedding_dev_ref[embedding_offset + emb_idx],
                                               embeddings);
    }
  }
}

template <typename InputT, typename IndexT, typename EmbeddingT>
void scatter_temp_func(const void* input,
                       wholememory_matrix_description_t input_desc,
                       void* indices,
                       int64_t indice_count,
                       wholememory_gref_t embedding_gref,
                       wholememory_matrix_description_t embedding_desc,
                       cudaStream_t stream)
{
  WHOLEMEMORY_EXPECTS(input_desc.sizes[1] == indice_count,
                      "scatter_func, input shape[1]=%ld, but indice_count=%ld",
                      input_desc.sizes[1],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[0] == 0) return;
  int wm_alignment   = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment   = determine_memory_alignment_elt_count(input, input_desc);
  int alignment      = std::min<int>(wm_alignment, mm_alignment);
  int embedding_size = embedding_desc.sizes[0];
  int thread_x       = raft::div_rounding_up_safe<int>(embedding_size, alignment);
  thread_x           = std::min(thread_x, 256);
  int thread_y       = 1;
  if (thread_x < 64) {
    int power2_thread_x = 1;
    for (; power2_thread_x < thread_x; power2_thread_x *= 2)
      ;
    thread_x = power2_thread_x;
    thread_y = 64 / thread_x;
  }
  int64_t block_count_64 = indice_count / thread_y;
  int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
  dim3 block_dim(thread_x, thread_y, 1);
  void (*kernel_fn)(const InputT*,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    wholememory_gref_t,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 16>;
      break;
    }
    case 8: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 8>;
      break;
    }
    case 4: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 4>;
      break;
    }
    case 2: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 2>;
      break;
    }
    case 1: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("scatter func alignment=%d.", alignment);
      return;
    }
  }
  kernel_fn<<<block_count, block_dim, 0, stream>>>(static_cast<const InputT*>(input),
                                                   input_desc,
                                                   static_cast<const IndexT*>(indices),
                                                   indice_count,
                                                   embedding_gref,
                                                   embedding_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace wholememory_ops