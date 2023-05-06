import pytest
from pylibwholegraph.torch.initialize import load_wholegraph_op_libraries
import torch
from pylibwholegraph.test_utils.test_comm import gen_csr_graph
import pylibwholegraph.torch.graph_ops as wg_ops
import math


def host_edge_weight_softmax(csr_row_ptr_tensor, edge_weight_tensor):
    row_num = csr_row_ptr_tensor.shape[0] - 1
    num_head = edge_weight_tensor.shape[1]
    output_edge_weight_softmax_tensor = torch.empty(
        (edge_weight_tensor.shape[0], edge_weight_tensor.shape[1]),
        dtype=edge_weight_tensor.dtype,
    )
    for row_id in range(row_num):
        start = csr_row_ptr_tensor[row_id]
        end = csr_row_ptr_tensor[row_id + 1]
        for head_id in range(num_head):
            max_value = -1.0
            for j in range(start, end):
                max_value = max(max_value, edge_weight_tensor[j][head_id])
            sum_exp = 0.0
            for j in range(start, end):
                sum_exp += math.exp(edge_weight_tensor[j][head_id] - max_value)
            for j in range(start, end):
                output_edge_weight_softmax_tensor[j][head_id] = (
                    math.exp(edge_weight_tensor[j][head_id] - max_value) / sum_exp
                )

    return output_edge_weight_softmax_tensor


def routine_func(**kwargs):
    load_wholegraph_op_libraries()
    target_node_count = kwargs["target_node_count"]
    neighbor_node_count = kwargs["neighbor_node_count"]
    edge_num = kwargs["edge_num"]
    feature_dtype = kwargs["feature_dtype"]
    num_head = kwargs["num_head"]
    assert neighbor_node_count >= target_node_count
    csr_row_ptr_tensor, _, _ = gen_csr_graph(
        target_node_count,
        edge_num,
        neighbor_node_count,
        csr_row_dtype=torch.int32,
        csr_col_dtype=torch.int32,
    )
    edge_weight_tensor = torch.rand((edge_num, num_head), dtype=feature_dtype)
    edge_weight_tensor_cuda = edge_weight_tensor.cuda()
    csr_row_ptr_tensor_cuda = csr_row_ptr_tensor.cuda()
    edge_weight_softmax_tensor_cuda = wg_ops.edge_weight_softmax_forward(
        csr_row_ptr_tensor_cuda, edge_weight_tensor_cuda
    )
    edge_weight_softmax_tensor = edge_weight_softmax_tensor_cuda.cpu()

    edge_weight_softmax_tensor_ref = host_edge_weight_softmax(
        csr_row_ptr_tensor, edge_weight_tensor
    )
    assert torch.allclose(edge_weight_softmax_tensor, edge_weight_softmax_tensor_ref)


@pytest.mark.parametrize("target_node_count", [101, 113])
@pytest.mark.parametrize("neighbor_node_count", [157, 1987])
@pytest.mark.parametrize("edge_num", [1001, 2302])
@pytest.mark.parametrize("feature_dtype", [torch.float32])
@pytest.mark.parametrize("num_head", [8, 32])
def test_edge_weight_softmax_forward(
    target_node_count, neighbor_node_count, edge_num, feature_dtype, num_head
):
    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0
    routine_func(
        target_node_count=target_node_count,
        neighbor_node_count=neighbor_node_count,
        edge_num=edge_num,
        feature_dtype=feature_dtype,
        num_head=num_head,
    )
