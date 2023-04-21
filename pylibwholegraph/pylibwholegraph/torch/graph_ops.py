import torch
import pylibwholegraph.binding.wholememory_binding as wmb
from .wholegraph_env import get_stream, wholememory_dtype_to_pytorch_dtype, TorchMemoryContext, get_wholegraph_env_fns, wrap_torch_tensor
from typing import Union


def append_unique(
        target_node_tensor: torch.Tensor,
        neighbor_node_tensor: torch.Tensor,
        need_neighbor_raw_to_unique: bool = False):
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
    

def spmm_no_weight_forward(
        csr_row_ptr_tensor: torch.Tensor,
        csr_col_ptr_tensor: torch.Tensor,
        feature_tensor: torch.Tensor,
        aggregator: torch.int64):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert feature_tensor.dim() == 2
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32

    output_feature_tensor = torch.empty((csr_row_ptr_tensor.shape[0] - 1, feature_tensor.shape[1]), device = 'cuda', dtype = feature_tensor.dtype)

    wmb.spmm_no_weight_forward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(feature_tensor),
        aggregator,
        wrap_torch_tensor(output_feature_tensor),
        get_stream())
    
    return output_feature_tensor

def spmm_no_weight_backward(
        csr_row_ptr_tensor: torch.Tensor,
        csr_col_ptr_tensor: torch.Tensor,
        input_grad_feature_tensor: torch.Tensor,
        input_cout: torch.int64,
        aggregator: torch.int64):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32

    output_grad_feature_tensor = torch.empty((input_cout, input_grad_feature_tensor.shape[1]), device = 'cuda', dtype = input_grad_feature_tensor.dtype)

    wmb.spmm_no_weight_backward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(input_grad_feature_tensor),
        aggregator,
        wrap_torch_tensor(output_grad_feature_tensor),
        get_stream())

    return output_grad_feature_tensor
        

def spadd_gat_forward(
        csr_row_ptr_tensor: torch.Tensor,
        csr_col_ptr_tensor: torch.Tensor,
        edge_weight_left_tensor: torch.Tensor,
        edge_weight_right_tensor: torch.Tensor):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32 
    assert edge_weight_left_tensor.dim() == 2
    assert edge_weight_right_tensor.dim() == 2
    assert edge_weight_right_tensor.shape[1] == edge_weight_left_tensor.shape[1]
    assert edge_weight_left_tensor.shape[0] == csr_row_ptr_tensor.shape[0] - 1

    output_score_tensor = torch.empty((csr_col_ptr_tensor.shape[0], edge_weight_left_tensor.shape[1]), device='cuda', dtype = edge_weight_left_tensor.dtype)
    wmb.spadd_gat_forward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(edge_weight_left_tensor),
        wrap_torch_tensor(edge_weight_right_tensor),
        wrap_torch_tensor(output_score_tensor),
        get_stream())
    return output_score_tensor

def spadd_gat_backward(
        csr_row_ptr_tensor: torch.Tensor,
        csr_col_ptr_tensor: torch.Tensor,
        grad_score_tensor: torch.Tensor,
        neighbor_node_count: torch.int64):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32 
    assert grad_score_tensor.dim() == 2
    assert grad_score_tensor.shape[0] == csr_col_ptr_tensor.shape[0]

    output_edge_weight_left_tensor = torch.empty((csr_row_ptr_tensor.shape[0] - 1, grad_score_tensor.shape[1]), device='cuda', dtype = grad_score_tensor.dtype)
    output_edge_weight_right_tensor = torch.empty((neighbor_node_count, grad_score_tensor.shape[1]), device='cuda', dtype=grad_score_tensor.dtype)

    wmb.spadd_gat_backward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(grad_score_tensor),
        wrap_torch_tensor(output_edge_weight_left_tensor),
        wrap_torch_tensor(output_edge_weight_right_tensor),
        get_stream())
    
    return output_edge_weight_left_tensor,output_edge_weight_right_tensor

    