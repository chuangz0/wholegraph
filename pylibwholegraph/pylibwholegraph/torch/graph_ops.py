import torch
import pylibwholegraph.binding.wholememory_binding as wmb
from .wholegraph_env import get_stream, wholememory_dtype_to_pytorch_dtype, TorchMemoryContext, get_wholegraph_env_fns, wrap_torch_tensor
from typing import Union


def append_unique(
        target_node_tensor: torch.Tensor,
        neighbor_node_tensor: torch.Tensor,
        need_neighbor_raw_to_unique: bool = True):
    assert target_node_tensor.dim() == 1
    assert neighbor_node_tensor.dim() == 1

    output_unique_node_context = TorchMemoryContext()
    output_unique_node_tensor_id = id(output_unique_node_context)
    output_neighbor_raw_to_unique_mapping_tensor = None
    if need_neighbor_raw_to_unique:
        output_neighbor_raw_to_unique_mapping_tensor = torch.empty(neighbor_node_tensor.shape[0], device = 'cuda', dtype = torch.int)
    
    wmb.append_unique(wrap_torch_tensor(target_node_tensor),
                      wrap_torch_tensor(neighbor_node_tensor),
                      output_unique_node_tensor_id,
                      wrap_torch_tensor(output_neighbor_raw_to_unique_mapping_tensor),
                      get_wholegraph_env_fns(),
                      get_stream())
    if need_neighbor_raw_to_unique:
        return output_unique_node_context.get_tensor(), output_neighbor_raw_to_unique_mapping_tensor
    else:
        return output_unique_node_context.get_tensor()
    
