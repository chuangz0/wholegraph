import pytest
import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.utils.multiprocess import multiprocess_run
from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm, load_wholegraph_op_libraries
from pylibwholegraph.torch.dlpack_utils import torch_import_from_dlpack
import torch
import random 
from functools import partial
from pylibwholegraph.test_utils.test_comm import gen_csr_graph, copy_host_1D_tensor_to_wholememory, host_get_sample_offset_tensor, host_sample_all_neighbors, int_to_wholememory_datatype, int_to_wholememory_location, int_to_wholememory_type
import pylibwholegraph.torch.graph_ops as wg_ops


def host_neighbor_raw_to_unique(unique_node_tensor, neighbor_node_tensor):
    output_neighbor_raw_to_unique = torch.empty((neighbor_node_tensor.size(0)), dtype=torch.int32)
    for i in range(neighbor_node_tensor.size(0)):
        neighbor_id = neighbor_node_tensor[i]
        output_neighbor_raw_to_unique[i] = torch.nonzero(unique_node_tensor == neighbor_id).item()
    return output_neighbor_raw_to_unique

def routine_func(**kwargs):
    load_wholegraph_op_libraries()
    target_node_count = kwargs['target_node_count']
    neighbor_node_count = kwargs['neighbor_node_count']
    target_node_dtype = kwargs['target_node_dtype']
    need_neighbor_raw_to_unique = kwargs['need_neighbor_raw_to_unique']
    target_node_tensor = torch.randperm(neighbor_node_count, dtype=target_node_dtype)[:target_node_count]
    neighbor_node_tensor = torch.randint(0, neighbor_node_count, (neighbor_node_count, ), dtype=target_node_dtype)
    
    target_node_tensor_cuda = target_node_tensor.cuda()
    neighbor_node_tensor_cuda = neighbor_node_tensor.cuda()
    output_unique_node_tensor_cuda = None
    output_neighbor_raw_to_unique_mapping_tensor_cuda = None
    if need_neighbor_raw_to_unique:
        output_unique_node_tensor_cuda, output_neighbor_raw_to_unique_mapping_tensor_cuda = wg_ops.append_unique(target_node_tensor_cuda, neighbor_node_tensor_cuda, need_neighbor_raw_to_unique = need_neighbor_raw_to_unique)
    else:
        output_unique_node_tensor_cuda = wg_ops.append_unique(target_node_tensor_cuda, neighbor_node_tensor_cuda, need_neighbor_raw_to_unique = need_neighbor_raw_to_unique)
    
    output_unique_node_tensor = output_unique_node_tensor_cuda.cpu()

    output_unique_node_tensor_ref = torch.unique(torch.cat((target_node_tensor, neighbor_node_tensor), 0), sorted=True)
    output_unique_node_tensor_sorted, _ = torch.sort(output_unique_node_tensor)
    assert torch.equal(output_unique_node_tensor_sorted, output_unique_node_tensor_ref)

    if need_neighbor_raw_to_unique:
        output_neighbor_raw_to_unique_mapping_tensor = output_neighbor_raw_to_unique_mapping_tensor_cuda.cpu()
        output_neighbor_raw_to_unique_mapping_tensor_ref = host_neighbor_raw_to_unique(output_unique_node_tensor, neighbor_node_tensor)
        assert torch.equal(output_neighbor_raw_to_unique_mapping_tensor, output_neighbor_raw_to_unique_mapping_tensor_ref)


@pytest.mark.parametrize('target_node_count', [10, 113])
@pytest.mark.parametrize('neighbor_node_count', [104, 1987])
@pytest.mark.parametrize('target_node_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('need_neighbor_raw_to_unique', [True, False])
def test_append_unique(target_node_count, neighbor_node_count, target_node_dtype, need_neighbor_raw_to_unique):
    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0
    routine_func(target_node_count=target_node_count, neighbor_node_count=neighbor_node_count, target_node_dtype=target_node_dtype, need_neighbor_raw_to_unique=need_neighbor_raw_to_unique)
    