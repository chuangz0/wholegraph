import pytest
import pylibwholegraph.binding.wholememory_binding as wmb
import multiprocessing as mp
from multiprocessing import Process
import torch
import torch.distributed as dist
import os
import numpy as np


# Run with:
# python3 -m pytest ../tests/pylibwholegraph/test_wholememory_binding.py -s

def multiprocess_run(world_size: int, func):
    assert world_size > 0
    # if world_size == 1:
    #    func(0, 1)
    #    return
    mp.set_start_method('spawn')
    process_array = [None] * world_size
    for i in range(world_size):
        process_array[i] = Process(target=func, args=(i, world_size))
        process_array[i].start()
    for i in range(world_size):
        process_array[i].join()
    for i in range(world_size):
        assert process_array[i].exitcode == 0


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
    uid_th = torch.from_dlpack(wm_uid)
    uid_th_cuda = uid_th.cuda()
    dist.broadcast(uid_th_cuda, 0)
    uid_th.copy_(uid_th_cuda.cpu())

    wm_comm = wmb.create_communicator(wm_uid, world_rank, world_size)
    return wm_comm


def torch_import_from_dlpack(dp):
    return torch.from_dlpack(dp)


def single_test_case(wm_comm, mt, ml, malloc_size, granularity):
    world_rank = wmb.comm_get_rank(wm_comm)
    world_size = wmb.comm_get_size(wm_comm)
    print('Rank=%d testing mt=%s, ml=%s' % (world_rank, mt, ml))
    h = wmb.malloc(malloc_size, wm_comm, mt, ml, granularity)
    global_tensor = None
    chunked_tensors = None
    view_device = wmb.WholeMemoryMemoryLocation.MlDevice
    view_device_id = world_rank
    tensor_data_type = wmb.WholeMemoryDataType.DtInt64
    elt_size = 8

    local_tensor, local_offset = wmb.get_local_tensor(h, torch_import_from_dlpack, tensor_data_type,
                                                      view_device, view_device_id)
    local_data_torch = torch.arange(local_offset, local_offset + local_tensor.shape[0], dtype=torch.int64)
    local_tensor.copy_(local_data_torch)

    local_view_tensor, _ = wmb.get_local_tensor(h, torch_import_from_dlpack, tensor_data_type,
                                                view_device, view_device_id)
    assert torch.equal(local_view_tensor.cpu(), local_data_torch) == True
    del local_data_torch, local_view_tensor

    wmb.comm_barrier(wm_comm)

    if mt == wmb.WholeMemoryMemoryType.MtDistributed or (
            mt == wmb.WholeMemoryMemoryType.MtChunked and ml == wmb.WholeMemoryMemoryLocation.MlDevice):
        with pytest.raises(ValueError):
            global_tensor = wmb.get_global_tensor(h, torch_import_from_dlpack, tensor_data_type,
                                                  view_device, view_device_id)
    else:
        global_tensor = wmb.get_global_tensor(h, torch_import_from_dlpack, tensor_data_type,
                                              view_device, view_device_id)
        global_data_torch = torch.arange(0, malloc_size // elt_size, dtype=torch.int64)
        assert torch.equal(global_tensor.cpu(), global_data_torch) == True
        del global_data_torch

    if mt == wmb.WholeMemoryMemoryType.MtDistributed:
        with pytest.raises(ValueError):
            chunked_tensors = wmb.get_all_chunked_tensor(h, torch_import_from_dlpack, tensor_data_type,
                                                         view_device, view_device_id)
    else:
        chunked_tensors = wmb.get_all_chunked_tensor(h, torch_import_from_dlpack, tensor_data_type,
                                                     view_device, view_device_id)
        remote_offset = 0
        for i in range(len(chunked_tensors)):
            remote_data_torch = torch.arange(remote_offset, remote_offset + chunked_tensors[i].shape[0], dtype=torch.int64)
            assert torch.equal(chunked_tensors[i].cpu(), remote_data_torch) == True
            remote_offset += chunked_tensors[i].shape[0]
            del remote_data_torch

    wmb.free(h)


def routine_func(world_rank: int, world_size: int):
    wm_comm = init_torch_env_and_create_wm_comm(world_rank, world_size)

    single_rank_size = 1024 * 1024 * 1024
    malloc_size = single_rank_size * world_size
    granularity = 256

    for mt in [wmb.WholeMemoryMemoryType.MtContinuous, wmb.WholeMemoryMemoryType.MtChunked,
               wmb.WholeMemoryMemoryType.MtDistributed]:
        for ml in [wmb.WholeMemoryMemoryLocation.MlHost, wmb.WholeMemoryMemoryLocation.MlDevice]:
            single_test_case(wm_comm, mt, ml, malloc_size, granularity)


def test_dlpack():
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    multiprocess_run(gpu_count, routine_func)
