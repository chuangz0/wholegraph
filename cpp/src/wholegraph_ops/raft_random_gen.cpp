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

#include <wholememory/wholegraph_op.h>
#include <cmath>
#include <wholememory_ops/raft_random.cuh>


#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t raft_pcg_generator_random_int(
  int64_t random_seed,
  int64_t subsequence,
  wholememory_tensor_t output
) {
  auto output_tensor_desc = *wholememory_tensor_get_tensor_description(output);
  if (output_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("output should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_tensor_desc.dtype != WHOLEMEMORY_DT_INT64) {
    WHOLEMEMORY_ERROR("output should be int64 tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }

  int64_t* output_ptr = static_cast<int64_t*>(wholememory_tensor_get_data_pointer(output));
  PCGenerator rng((unsigned long long)random_seed, subsequence, 0);
  for (int64_t i = 0; i < output_tensor_desc.sizes[0]; i++) {
    uint32_t random_num;
    rng.next(random_num);
    output_ptr[i] = static_cast<int64_t>(random_num);
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t raft_pcg_generator_random_float(
  int64_t random_seed,
  int64_t subsequence,
  wholememory_tensor_t output
) {
  auto output_tensor_desc = *wholememory_tensor_get_tensor_description(output);
  if (output_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("output should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_tensor_desc.dtype != WHOLEMEMORY_DT_FLOAT) {
    WHOLEMEMORY_ERROR("output should be float tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* output_ptr = static_cast<float*>(wholememory_tensor_get_data_pointer(output));
  PCGenerator rng((unsigned long long)random_seed, subsequence, 0);
  for (int64_t i = 0; i < output_tensor_desc.sizes[0]; i++) {
    float random_num;
    random_num = rng.next_float();
    output_ptr[i] = static_cast<int64_t>(random_num);
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t raft_pcg_generator_random_float_with_bias(
  int64_t random_seed,
  int64_t subsequence,
  wholememory_tensor_t weight,
  wholememory_tensor_t output
) {
  auto output_tensor_desc = *wholememory_tensor_get_tensor_description(output);
  auto weight_tensor_desc = *wholememory_tensor_get_tensor_description(weight);
  if (output_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("output should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (output_tensor_desc.dtype != weight_tensor_desc.dtype) {
    WHOLEMEMORY_ERROR("output tensor dtype should be the same with weight tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (weight_tensor_desc.dim != 1) {
     WHOLEMEMORY_ERROR("weight should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* output_ptr = wholememory_tensor_get_data_pointer(output);
  auto* weight_ptr = wholememory_tensor_get_data_pointer(weight);
  PCGenerator rng((unsigned long long)random_seed, subsequence, 0);
  for (int64_t i = 0; i < output_tensor_desc.sizes[0]; i++) {
     float u             = -rng.next_float(1.0f, 0.5f);
    int64_t random_num2 = 0;
    int seed_count      = -1;
    do {
      rng.next(random_num2);
      seed_count++;
    } while (!random_num2);
    auto count_one = [](unsigned long long num) {
      int c = 0;
      while (num) {
        num >>= 1;
        c++;
      }
      return 64 - c;
    };
    int one_bit = count_one(random_num2) + seed_count * 64;
    u *= pow(2, -one_bit);
    // float logk = (log1pf(u) / logf(2.0)) * (1.0f / (float)weight);
    if (weight_tensor_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
      float weight_value                   = static_cast<float*>(weight_ptr)[i];
      float logk                     = (1 / weight_value) * (log1p(u) / log(2.0));
      static_cast<float*>(output_ptr)[i] = logk;
    } else if (weight_tensor_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
      double weight_value                   = static_cast<double*>(weight_ptr)[i];
      double logk                     = (1 / weight_value) * (log1p(u) / log(2.0));
      static_cast<double*>(output_ptr)[i] = logk;
    }
  }
  return WHOLEMEMORY_SUCCESS;
}