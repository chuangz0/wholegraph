import pytest
import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.utils.multiprocess import multiprocess_run
from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm, load_wholegraph_op_libraries
from pylibwholegraph.torch.dlpack_utils import torch_import_from_dlpack
import torch
import random 
from functools import partial
from pylibwholegraph.test_utils.test_comm import gen_csr_graph, copy_host_1D_tensor_to_wholememory, host_get_sample_offset_tensor, host_sample_all_neighbors, int_to_wholememory_datatype, int_to_wholememory_location, int_to_wholememory_type

def  host_weighted_sample_without_replacement_func(host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr, center_nodes, output_sample_offset_tensor, col_id_dtype, csr_weight_dtype, total_sample_count, max_sample_count, random_seed):
    output_dest_tensor = torch.empty((total_sample_count,), dtype = col_id_dtype)
    output_center_localid_tensor = torch.empty((total_sample_count,), dtype = torch.int32)
    output_edge_gid_tensor = torch.empty((total_sample_count,), dtype = torch.int64)
    center_nodes_count = center_nodes.size(0)
    block_sizes = [128, 256, 256, 512]
    items_per_threads = [4, 4, 8, 8]
    fun_idx = int((max_sample_count - 1) / 128)
    if fun_idx > 3:
        fun_idx = 3
    
    block_size = block_sizes[fun_idx]
    items_per_thread = items_per_threads[fun_idx]

    for i in range(center_nodes_count):
        node_id = center_nodes[i]
        start = host_csr_row_ptr[node_id]
        end = host_csr_row_ptr[node_id + 1]
        neighbor_count = end - start
        output_id = output_sample_offset_tensor[i]
        gidx = i * block_size
        if (neighbor_count <= max_sample_count):
            for j in range(end - start):
                output_dest_tensor[output_id + j] = host_csr_col_ptr[start + j]
                output_center_localid_tensor[output_id + j] = i
                output_edge_gid_tensor[output_id + j] = start + j 
        else:
            total_neighbor_generated_weights = torch.tensor([], dtype = csr_weight_dtype)
            edge_weight_corresponding_ids = torch.tensor([], dtype = col_id_dtype)
            for j in range(block_size):
                local_gidx = gidx + j
                local_edge_weights = torch.empty((items_per_thread, ), dtype = csr_weight_dtype)
                generated_edge_weight_count = 0
                for k in range(items_per_thread):
                    id = k * block_size + j
                    if id < neighbor_count:
                        local_edge_weights[k] = host_csr_weight_ptr[start + id]
                        generated_edge_weight_count += 1
                        edge_weight_corresponding_ids = torch.cat((edge_weight_corresponding_ids, torch.tensor([id], dtype=col_id_dtype)))
                generated_random_weight = torch.ops.wholegraph_test.raft_pcg_generator_random_from_weight(random_seed, local_gidx, local_edge_weights, generated_edge_weight_count)
                total_neighbor_generated_weights = torch.cat((total_neighbor_generated_weights, generated_random_weight))
            assert total_neighbor_generated_weights.size(0) == neighbor_count
            _, sorted_weight_ids = torch.sort(total_neighbor_generated_weights, descending=True)
            sorted_top_m_weight_ids = edge_weight_corresponding_ids[sorted_weight_ids[0:max_sample_count]]
            for sample_id in range(max_sample_count):
                output_dest_tensor[output_id + sample_id] = host_csr_col_ptr[start + sorted_top_m_weight_ids[sample_id]]
                output_center_localid_tensor[output_id + sample_id] = i
                output_edge_gid_tensor[output_id + sample_id] = start + sorted_top_m_weight_ids[sample_id]
    return output_dest_tensor, output_center_localid_tensor, output_edge_gid_tensor


def  host_weighted_sample_without_replacement(host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr, center_nodes, max_sample_count, col_id_dtype, random_seed):
    center_nodes_count = center_nodes.size(0)
    output_sample_offset_tensor = host_get_sample_offset_tensor(host_csr_row_ptr, center_nodes, max_sample_count)
    total_sample_count = output_sample_offset_tensor[center_nodes_count]

    if max_sample_count <= 0:
        return host_sample_all_neighbors(host_csr_row_ptr, host_csr_col_ptr, center_nodes, output_sample_offset_tensor, col_id_dtype, total_sample_count)
    if max_sample_count > 1024:
        raise ValueError("invalid host_unweighted_sample_without_replacement test max_sample_count")
    

    torch_col_id_dtype = torch.int32
    if (col_id_dtype == wmb.WholeMemoryDataType.DtInt64):
        torch_col_id_dtype = torch.int64 

    output_dest_tensor, output_center_localid_tensor, output_edge_gid_tensor = host_weighted_sample_without_replacement_func(host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr, center_nodes, output_sample_offset_tensor, torch_col_id_dtype, host_csr_weight_ptr.dtype, total_sample_count, max_sample_count, random_seed)

    return output_sample_offset_tensor, output_dest_tensor, output_center_localid_tensor, output_edge_gid_tensor
  

def routine_func(world_rank: int, world_size: int, **kwargs):
    wm_comm = init_torch_env_and_create_wm_comm(world_rank, world_size)
    load_wholegraph_op_libraries()
    host_csr_row_ptr = kwargs['host_csr_row_ptr']
    host_csr_col_ptr = kwargs['host_csr_col_ptr']
    host_csr_weight_ptr = kwargs['host_csr_weight_ptr']
    graph_node_count = kwargs['graph_node_count']
    graph_edge_count = kwargs['graph_edge_count']
    max_sample_count = kwargs['max_sample_count']
    center_node_count = kwargs['center_node_count']
    center_node_dtype = kwargs['center_node_dtype']
    int_col_id_dtype = kwargs['col_id_dtype']
    int_csr_weight_dtype = kwargs['csr_weight_dtype']
    int_wholememory_location = kwargs['wholememory_location']
    int_wholememory_type = kwargs['wholememory_type']

    world_rank = wm_comm.get_rank()
    world_size = wm_comm.get_size()

    col_id_dtype = int_to_wholememory_datatype(int_col_id_dtype)
    csr_weight_dtype = int_to_wholememory_datatype(int_csr_weight_dtype)
    wholememory_location = int_to_wholememory_location(int_wholememory_location)
    wholememory_type = int_to_wholememory_type(int_wholememory_type)


    wm_csr_row_ptr = wmb.create_wholememory_array(wmb.WholeMemoryDataType.DtInt64, graph_node_count + 1, wm_comm, wholememory_type, wholememory_location)
    wm_csr_col_ptr = wmb.create_wholememory_array(col_id_dtype, graph_edge_count, wm_comm, wholememory_type, wholememory_location)
    wm_csr_weight_ptr = wmb.create_wholememory_array(csr_weight_dtype, graph_edge_count, wm_comm, wholememory_type, wholememory_location)

    copy_host_1D_tensor_to_wholememory(wm_csr_row_ptr, host_csr_row_ptr, world_rank, world_size)
    copy_host_1D_tensor_to_wholememory(wm_csr_col_ptr, host_csr_col_ptr, world_rank, world_size)
    copy_host_1D_tensor_to_wholememory(wm_csr_weight_ptr, host_csr_weight_ptr, world_rank, world_size)

    wm_comm.barrier()

    center_node_tensor = torch.randint(0, graph_node_count, (center_node_count,), dtype = center_node_dtype)
    center_node_tensor_cuda = center_node_tensor.cuda()
    random_seed = random.randint(1, 10000)

    output_sample_offset_tensor_cuda, output_dest_tensor_cuda, output_center_localid_tensor_cuda, output_edge_gid_tensor_cuda = torch.ops.wholegraph.weighted_sample_without_replacement(wm_csr_row_ptr.get_c_handle(),
                                wm_csr_col_ptr.get_c_handle(),
                                wm_csr_weight_ptr.get_c_handle(),
                                center_node_tensor_cuda,
                                max_sample_count,
                                random_seed) 
    output_sample_offset_tensor = output_sample_offset_tensor_cuda.cpu()
    output_dest_tensor = output_dest_tensor_cuda.cpu()
    output_center_localid_tensor = output_center_localid_tensor_cuda.cpu()
    output_edge_gid_tensor = output_edge_gid_tensor_cuda.cpu()

    output_sample_offset_tensor_ref, output_dest_tensor_ref, output_center_localid_tensor_ref, output_edge_gid_tensor_ref = host_weighted_sample_without_replacement(host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr, center_node_tensor, max_sample_count, col_id_dtype, random_seed)
    
    assert torch.equal(output_sample_offset_tensor, output_sample_offset_tensor_ref)

    for i in range(center_node_count):
        start = output_sample_offset_tensor[i]
        end = output_sample_offset_tensor[i + 1]
        output_dest_tensor[start:end], sorted_ids = torch.sort(output_dest_tensor[start:end])
        output_center_localid_tensor[start:end] = output_center_localid_tensor[start:end][sorted_ids]
        output_edge_gid_tensor[start:end] = output_edge_gid_tensor[start:end][sorted_ids]

        output_dest_tensor_ref[start:end], ref_sorted_ids = torch.sort(output_dest_tensor_ref[start:end])
        output_center_localid_tensor_ref[start:end] = output_center_localid_tensor_ref[start:end][ref_sorted_ids]
        output_edge_gid_tensor_ref[start:end] = output_edge_gid_tensor_ref[start:end][ref_sorted_ids]
    

    assert torch.equal(output_dest_tensor, output_dest_tensor_ref)
    assert torch.equal(output_center_localid_tensor, output_center_localid_tensor_ref)
    assert torch.equal(output_edge_gid_tensor, output_edge_gid_tensor_ref)

    wmb.destroy_wholememory_tensor(wm_csr_row_ptr)
    wmb.destroy_wholememory_tensor(wm_csr_col_ptr)



@pytest.mark.parametrize('graph_node_count', [1103])
@pytest.mark.parametrize('graph_edge_count', [10437])
@pytest.mark.parametrize('max_sample_count', [119])
@pytest.mark.parametrize('center_node_count', [137])
@pytest.mark.parametrize('center_node_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('col_id_dtype', [0, 1])
@pytest.mark.parametrize('csr_weight_dtype', [2, 3])
@pytest.mark.parametrize('wholememory_location', ([0, 1]))
@pytest.mark.parametrize('wholememory_type', ([0, 1]))
def test_wholegraph_weighted_sample(graph_node_count, graph_edge_count, max_sample_count, center_node_count, center_node_dtype, col_id_dtype, csr_weight_dtype, wholememory_location, wholememory_type):
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    csr_col_dtype = torch.int32
    if (col_id_dtype == 1):
        csr_col_dtype = torch.int64 
    host_csr_row_ptr, host_csr_col_ptr, host_csr_weight_ptr = gen_csr_graph(graph_node_count, graph_edge_count, csr_col_dtype=csr_col_dtype)
    routine_func_partial = partial(routine_func, host_csr_row_ptr = host_csr_row_ptr, host_csr_col_ptr = host_csr_col_ptr, host_csr_weight_ptr = host_csr_weight_ptr, graph_node_count = graph_node_count, graph_edge_count = graph_edge_count, 
                                   max_sample_count = max_sample_count, center_node_count = center_node_count, center_node_dtype = center_node_dtype, col_id_dtype = col_id_dtype, csr_weight_dtype = csr_weight_dtype,
                                   wholememory_location = wholememory_location,  wholememory_type = wholememory_type)
    multiprocess_run(gpu_count, routine_func_partial, True)