# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import pylibwholegraph.binding.wholememory_binding as wmb
from .wholegraph_env import (
    get_stream,
    TorchMemoryContext,
    get_wholegraph_env_fns,
    wrap_torch_tensor,
)
from typing import Union
import random


def unweighted_sample_without_replacement(
    wm_csr_row_ptr_tensor: wmb.PyWholeMemoryTensor,
    wm_csr_col_ptr_tensor: wmb.PyWholeMemoryTensor,
    center_nodes_tensor: torch.Tensor,
    max_sample_count: int,
    random_seed: Union[int, None] = None,
    need_center_local_output: bool = False,
    need_edge_output: bool = False,
):
    assert wm_csr_row_ptr_tensor.dim() == 1
    assert wm_csr_col_ptr_tensor.dim() == 1
    assert center_nodes_tensor.dim() == 1
    if random_seed is None:
        random_seed = random.getrandbits(64)
    output_sample_offset_tensor = torch.empty(
        center_nodes_tensor.shape[0] + 1, device="cuda", dtype=torch.int
    )
    output_dest_context = TorchMemoryContext()
    output_dest_tensor_id = id(output_dest_context)
    output_center_localid_context = None
    output_center_localid_tensor_id = 0
    output_edge_gid_context = None
    output_edge_gid_tensor_id = 0
    if need_center_local_output:
        output_center_localid_context = TorchMemoryContext()
        output_center_localid_tensor_id = id(output_center_localid_context)
    if need_edge_output:
        output_edge_gid_context = TorchMemoryContext()
        output_edge_gid_tensor_id = id(output_edge_gid_context)
    wmb.csr_unweighted_sample_without_replacement(
        wm_csr_row_ptr_tensor,
        wm_csr_col_ptr_tensor,
        wrap_torch_tensor(center_nodes_tensor),
        max_sample_count,
        wrap_torch_tensor(output_sample_offset_tensor),
        output_dest_tensor_id,
        output_center_localid_tensor_id,
        output_edge_gid_tensor_id,
        random_seed,
        get_wholegraph_env_fns(),
        get_stream(),
    )
    if need_edge_output and need_center_local_output:
        return (
            output_sample_offset_tensor,
            output_dest_context.get_tensor(),
            output_center_localid_context.get_tensor(),
            output_edge_gid_context.get_tensor(),
        )
    elif need_center_local_output:
        return (
            output_sample_offset_tensor,
            output_dest_context.get_tensor(),
            output_center_localid_context.get_tensor(),
        )
    elif need_edge_output:
        return (
            output_sample_offset_tensor,
            output_dest_context.get_tensor(),
            output_edge_gid_context.get_tensor(),
        )
    else:
        return output_sample_offset_tensor, output_dest_context.get_tensor()


def weighted_sample_without_replacement(
    wm_csr_row_ptr_tensor: wmb.PyWholeMemoryTensor,
    wm_csr_col_ptr_tensor: wmb.PyWholeMemoryTensor,
    wm_csr_weight_ptr_tensor: wmb.PyWholeMemoryTensor,
    center_nodes_tensor: torch.Tensor,
    max_sample_count: int,
    random_seed: Union[int, None] = None,
    need_center_local_output: bool = False,
    need_edge_output: bool = False,
):
    assert wm_csr_row_ptr_tensor.dim() == 1
    assert wm_csr_col_ptr_tensor.dim() == 1
    assert wm_csr_weight_ptr_tensor.dim() == 1
    assert wm_csr_weight_ptr_tensor.shape[0] == wm_csr_col_ptr_tensor.shape[0]
    assert center_nodes_tensor.dim() == 1
    if random_seed is None:
        random_seed = random.getrandbits(64)
    output_sample_offset_tensor = torch.empty(
        center_nodes_tensor.shape[0] + 1, device="cuda", dtype=torch.int
    )
    output_dest_context = TorchMemoryContext()
    output_dest_tensor_id = id(output_dest_context)
    output_center_localid_context = None
    output_center_localid_tensor_id = 0
    output_edge_gid_context = None
    output_edge_gid_tensor_id = 0
    if need_center_local_output:
        output_center_localid_context = TorchMemoryContext()
        output_center_localid_tensor_id = id(output_center_localid_context)
    if need_edge_output:
        output_edge_gid_context = TorchMemoryContext()
        output_edge_gid_tensor_id = id(output_edge_gid_context)
    wmb.csr_weighted_sample_without_replacement(
        wm_csr_row_ptr_tensor,
        wm_csr_col_ptr_tensor,
        wm_csr_weight_ptr_tensor,
        wrap_torch_tensor(center_nodes_tensor),
        max_sample_count,
        wrap_torch_tensor(output_sample_offset_tensor),
        output_dest_tensor_id,
        output_center_localid_tensor_id,
        output_edge_gid_tensor_id,
        random_seed,
        get_wholegraph_env_fns(),
        get_stream(),
    )
    if need_edge_output and need_center_local_output:
        return (
            output_sample_offset_tensor,
            output_dest_context.get_tensor(),
            output_center_localid_context.get_tensor(),
            output_edge_gid_context.get_tensor(),
        )
    elif need_center_local_output:
        return (
            output_sample_offset_tensor,
            output_dest_context.get_tensor(),
            output_center_localid_context.get_tensor(),
        )
    elif need_edge_output:
        return (
            output_sample_offset_tensor,
            output_dest_context.get_tensor(),
            output_edge_gid_context.get_tensor(),
        )
    else:
        return output_sample_offset_tensor, output_dest_context.get_tensor()


def random_int_generator(random_seed,
                         sub_sequence,
                         output_random_value_count):
    output = torch.empty((output_random_value_count,), dtype=torch.int)
    wmb.raft_generator_random_int(random_seed, sub_sequence, wrap_torch_tensor(output))
    return output

def random_float_generator(random_seed: int,
                           sub_sequence: int,
                           output_random_value_count: int):
    output = torch.empty((output_random_value_count,), dtype = torch.float)
    wmb.raft_generator_random_float(random_seed, sub_sequence, wrap_torch_tensor(output))
    return output

def random_float_generator_with_bias(weight: torch.tensor,
                                     input_radom_values: torch.tensor,
                                     output_random_value_count: int):
    output = torch.empty((output_random_value_count,), dtype=weight.dtype)
    assert weight.dtype == input_radom_values.dtype
    wmb.raft_generator_random_float_with_bias(wrap_torch_tensor(weight), wrap_torch_tensor(input_radom_values), wrap_torch_tensor(output))
    return output
