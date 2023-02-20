import torch
import torch.utils.dlpack


def torch_import_from_dlpack(dp):
    return torch.utils.dlpack.from_dlpack(dp.__dlpack__())

