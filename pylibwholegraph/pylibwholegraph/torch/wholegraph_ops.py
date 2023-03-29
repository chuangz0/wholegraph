import torch
import pylibwholegraph.binding.wholememory_binding as wmb
from .wholegraph_env import get_stream, wholememory_dtype_to_pytorch_dtype, TorchMemoryContext, get_wholegraph_env_fns, wrap_torch_tensor
from typing import Union
import random

def unweighted_sample_without_replacement(
        wm_csr_row_ptr_tensor: wmb.PyWholeMemoryTensor,
        wm_csr_col_ptr_tensor: wmb.PyWholeMemoryTensor,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        random_seed: Union[int, None] = None,
        need_edge_output: bool = True):
    assert wm_csr_row_ptr_tensor.dim() == 1
    assert wm_csr_col_ptr_tensor.dim() == 1
    assert center_nodes_tensor.dim() == 1
    if random_seed is None:
        random_seed = random.getrandbits(64)
    output_sample_offset_tensor = torch.empty(center_nodes_tensor.shape[0] + 1, device='cuda', dtype=torch.int)
    output_dest_context = TorchMemoryContext()
    output_dest_tensor_id = id(output_dest_context)
    output_center_localid_context = TorchMemoryContext()
    output_center_localid_tensor_id = id(output_center_localid_context)
    output_edge_gid_context = None
    output_edge_gid_tensor_id = 0
    if need_edge_output:
        output_edge_gid_context = TorchMemoryContext()
        output_edge_gid_tensor_id = id(output_edge_gid_context)
    wmb.csr_unweighted_sample_without_replacement(wm_csr_row_ptr_tensor,
                                                  wm_csr_col_ptr_tensor,
                                                  wrap_torch_tensor(center_nodes_tensor),
                                                  max_sample_count,
                                                  wrap_torch_tensor(output_sample_offset_tensor),
                                                  output_dest_tensor_id,
                                                  output_center_localid_tensor_id,
                                                  output_edge_gid_tensor_id,
                                                  random_seed,
                                                  get_wholegraph_env_fns(),
                                                  get_stream())
    if need_edge_output:
        return output_sample_offset_tensor, output_dest_context.get_tensor(), output_center_localid_context.get_tensor(), output_edge_gid_context.get_tensor()
    else:
        return output_sample_offset_tensor, output_dest_context.get_tensor(), output_center_localid_context.get_tensor()


def weighted_sample_without_replacement(
        wm_csr_row_ptr_tensor: wmb.PyWholeMemoryTensor,
        wm_csr_col_ptr_tensor: wmb.PyWholeMemoryTensor,
        wm_csr_weight_ptr_tensor: wmb.PyWholeMemoryTensor,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        random_seed: Union[int, None] = None,
        need_edge_output: bool = True):
    assert wm_csr_row_ptr_tensor.dim() == 1
    assert wm_csr_col_ptr_tensor.dim() == 1
    assert wm_csr_weight_ptr_tensor.dim() == 1
    assert wm_csr_weight_ptr_tensor.shape[0] == wm_csr_col_ptr_tensor.shape[0]
    assert center_nodes_tensor.dim() == 1
    if random_seed is None:
        random_seed = random.getrandbits(64)
    output_sample_offset_tensor = torch.empty(center_nodes_tensor.shape[0] + 1, device='cuda', dtype=torch.int)
    output_dest_context = TorchMemoryContext()
    output_dest_tensor_id = id(output_dest_context)
    output_center_localid_context = TorchMemoryContext()
    output_center_localid_tensor_id = id(output_center_localid_context)
    output_edge_gid_context = None
    output_edge_gid_tensor_id = 0
    if need_edge_output:
        output_edge_gid_context = TorchMemoryContext()
        output_edge_gid_tensor_id = id(output_edge_gid_context)
    wmb.csr_weighted_sample_without_replacement(wm_csr_row_ptr_tensor,
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
                                                get_stream())
    if need_edge_output:
        return output_sample_offset_tensor, output_dest_context.get_tensor(), output_center_localid_context.get_tensor(), output_edge_gid_context.get_tensor()
    else:
        return output_sample_offset_tensor, output_dest_context.get_tensor(), output_center_localid_context.get_tensor()
