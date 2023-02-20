import os
import torch
import torch.distributed as dist
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb


def init_torch_env_and_create_wm_comm(world_rank: int, world_size: int):
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = str(12343)

    wmb.init(0)
    torch.set_num_threads(1)
    torch.cuda.set_device(world_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    if world_rank == 0:
        wm_uid = wmb.create_unique_id()
    else:
        wm_uid = wmb.PyWholeMemoryUniqueID()
    uid_th = torch.utils.dlpack.from_dlpack(wm_uid.__dlpack__())
    uid_th_cuda = uid_th.cuda()
    dist.broadcast(uid_th_cuda, 0)
    uid_th.copy_(uid_th_cuda.cpu())

    wm_comm = wmb.create_communicator(wm_uid, world_rank, world_size)
    return wm_comm


def load_wholegraph_op_libraries():
    cxx11abi = torch.torch.compiled_with_cxx11_abi()
    if cxx11abi is True:
        lib_path = 'wholegraph_torch/libwholegraph_torch_cxx11abi.so'
    else:
        lib_path = 'wholegraph_torch/libwholegraph_torch_precxx11abi.so'
    torch.ops.load_library(lib_path)


def jit_load_wholegraph_op_libraries():
    pass
