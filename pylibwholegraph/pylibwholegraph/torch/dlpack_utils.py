import torch


def torch_import_from_dlpack(dp):
    return torch.from_dlpack(dp)


