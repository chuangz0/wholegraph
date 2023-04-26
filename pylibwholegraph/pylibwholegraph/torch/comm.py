import os
import torch
import torch.distributed as dist
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb
from . import utils


class WholeMemoryCommunicator(object):
    r"""WholeMemory Communicator
    """
    def __init__(self, wmb_comm: wmb.PyWholeMemoryComm):
        super().__init__()
        self.wmb_comm = wmb_comm

    def get_rank(self):
        return self.wmb_comm.get_rank()

    def get_size(self):
        return self.wmb_comm.get_size()

    def barrier(self):
        return self.wmb_comm.barrier()

    def destroy(self):
        wmb.destroy_communicator(self.wmb_comm)
        self.wmb_comm = None


def create_group_communicator(group_size: int = -1, comm_stride: int = 1):
    r"""Create WholeMemory Communicator.
    For example: 24 ranks with group_size = 4 and comm_stride = 2 will create following groups:
    [0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23]
    :param group_size: Size of each group, -1 means to use all ranks in just one single group.
    :param comm_stride: Stride of each rank in each group
    :return: WholeMemoryCommunicator
    """
    world_size = dist.get_world_size()
    if group_size == -1:
        group_size = world_size
    strided_group_size = group_size * comm_stride
    assert world_size % strided_group_size == 0
    strided_group_count = world_size // strided_group_size
    world_rank = dist.get_rank()
    strided_group_idx = world_rank // strided_group_size
    idx_in_strided_group = world_rank % strided_group_size
    inner_group_idx = idx_in_strided_group % comm_stride
    idx_in_group = idx_in_strided_group // comm_stride
    wm_uid = wmb.PyWholeMemoryUniqueID()
    for strided_group in range(strided_group_count):
        for inner_group in range(comm_stride):
            group_root_rank = strided_group * strided_group_size + inner_group
            if world_rank == group_root_rank:
                tmp_wm_uid = wmb.create_unique_id()
            else:
                tmp_wm_uid = wmb.PyWholeMemoryUniqueID()
            uid_th = torch.utils.dlpack.from_dlpack(tmp_wm_uid.__dlpack__())
            uid_th_cuda = uid_th.cuda()
            dist.broadcast(uid_th_cuda, group_root_rank)
            uid_th.copy_(uid_th_cuda.cpu())
            if strided_group_idx == strided_group and inner_group_idx == inner_group:
                wm_uid_th = torch.utils.dlpack.from_dlpack(wm_uid.__dlpack__())
                wm_uid_th.copy_(uid_th)
    wm_comm = wmb.create_communicator(wm_uid, idx_in_group, group_size)
    return WholeMemoryCommunicator(wm_comm)


def destroy_communicator(wm_comm: WholeMemoryCommunicator):
    if wm_comm is not None and wm_comm.wmb_comm is not None:
        wmb.destroy_communicator(wm_comm.wmb_comm)
        wm_comm.wmb_comm = None
