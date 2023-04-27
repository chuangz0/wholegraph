import os
import torch
import torch.distributed as dist
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb
from . import utils
from .comm import WholeMemoryCommunicator, create_group_communicator, set_world_info
from .comm import get_global_communicator, get_local_node_communicator, get_local_device_communicator


def init_torch_env(world_rank: int, world_size: int, local_rank: int, local_size: int):
    r"""Init WholeGraph environment for PyTorch.
    :param world_rank: world rank of current process
    :param world_size: world size of all processes
    :param local_rank: local rank of current process
    :param local_size: local size
    :return: None
    """
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = str(12343)

    wmb.init(0)
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    set_world_info(world_rank, world_size, local_rank, local_size)


def init_torch_env_and_create_wm_comm(world_rank: int, world_size: int, local_rank: int, local_size: int):
    r"""Init WholeGraph environment for PyTorch and create single communicator for all ranks.
    :param world_rank: world rank of current process
    :param world_size: world size of all processes
    :param local_rank: local rank of current process
    :param local_size: local size
    :return: global and local node Communicator
    """
    init_torch_env(world_rank, world_size, local_rank, local_size)
    global_comm = get_global_communicator()
    local_comm = get_local_node_communicator()

    return global_comm, local_comm


def finalize():
    r"""Finalize WholeGraph.
    :return: None
    """
    wmb.finalize()


def load_wholegraph_op_libraries():
    cxx11abi = torch.torch.compiled_with_cxx11_abi()
    if cxx11abi is True:
        lib_path = 'wholegraph_torch/libwholegraph_torch_cxx11abi.so'
    else:
        lib_path = 'wholegraph_torch/libwholegraph_torch_precxx11abi.so'
    torch.ops.load_library(lib_path)


def jit_load_wholegraph_op_libraries():
    pass
