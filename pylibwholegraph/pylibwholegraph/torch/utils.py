import pylibwholegraph.binding.wholememory_binding as wmb
import torch
import os


WholeMemoryDataType = wmb.WholeMemoryDataType


def torch_dtype_to_wholememory_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float:
        return WholeMemoryDataType.DtFloat
    elif torch_dtype == torch.half:
        return WholeMemoryDataType.DtHalf
    elif torch_dtype == torch.double:
        return WholeMemoryDataType.DtDouble
    elif torch_dtype == torch.bfloat16:
        return WholeMemoryDataType.DtBF16
    elif torch_dtype == torch.int:
        return WholeMemoryDataType.DtInt
    elif torch_dtype == torch.int64:
        return WholeMemoryDataType.DtInt64
    elif torch_dtype == torch.int16:
        return WholeMemoryDataType.DtInt16
    elif torch_dtype == torch.int8:
        return WholeMemoryDataType.DtInt8
    else:
        raise ValueError('torch_dtype: %s not supported' % (torch_dtype, ))


def wholememory_dtype_to_torch_dtype(wm_dtype: WholeMemoryDataType):
    if wm_dtype == WholeMemoryDataType.DtFloat:
        return torch.float
    elif wm_dtype == WholeMemoryDataType.DtHalf:
        return torch.half
    elif wm_dtype == WholeMemoryDataType.DtDouble:
        return torch.double
    elif wm_dtype == WholeMemoryDataType.DtBF16:
        return torch.bfloat16
    elif wm_dtype == WholeMemoryDataType.DtInt:
        return torch.int
    elif wm_dtype == WholeMemoryDataType.DtInt64:
        return torch.int64
    elif wm_dtype == WholeMemoryDataType.DtInt16:
        return torch.int16
    elif wm_dtype == WholeMemoryDataType.DtInt8:
        return torch.int8
    else:
        raise ValueError('WholeMemoryMemory: %s not supported' % (int(wm_dtype),))


def get_file_size(filename: str):
    if not os.path.isfile(filename):
        raise ValueError('File %s not found or not file' % (filename,))
    if not os.access(filename, os.R_OK):
        raise ValueError('File %s not readable' % (filename,))
    file_size = os.path.getsize(filename)
    return file_size


def str_to_wmb_wholememory_memory_type(str_wmb_type: str):
    if str_wmb_type == 'continuous':
        return wmb.WholeMemoryMemoryType.MtContinuous
    elif str_wmb_type == 'chunked':
        return wmb.WholeMemoryMemoryType.MtChunked
    elif str_wmb_type == 'distributed':
        return wmb.WholeMemoryMemoryType.MtDistributed
    else:
        raise ValueError('WholeMemory type %s not supported, should be (continuous, chunked, distributed)'
                         % (str_wmb_type, ))


def str_to_wmb_wholememory_location(str_wmb_location: str):
    if str_wmb_location == 'cuda':
        return wmb.WholeMemoryMemoryLocation.MlDevice
    elif str_wmb_location == 'cpu':
        return wmb.WholeMemoryMemoryLocation.MlHost
    else:
        raise ValueError('WholeMemory location %s not supported, should be (cuda, cpu)' % (str_wmb_location, ))


def str_to_wmb_wholememory_access_type(str_wmb_access: str):
    if str_wmb_access == 'readonly' or str_wmb_access == 'ro':
        return wmb.WholeMemoryAccessType.AtReadOnly
    elif str_wmb_access == 'readwrite' or str_wmb_access == 'rw':
        return wmb.WholeMemoryAccessType.AtReadWrite
    else:
        raise ValueError('WholeMemory access %s not supported, should be (readonly, ro, readwrite, rw)'
                         % (str_wmb_access,))


def str_to_wmb_wholememory_optimizer_type(str_wmb_optimizer: str):
    if str_wmb_optimizer == 'sgd':
        return wmb.WholeMemoryOptimizerType.OptSgd
    elif str_wmb_optimizer == 'adam':
        return wmb.WholeMemoryOptimizerType.OptLazyAdam
    elif str_wmb_optimizer == 'adagrad':
        return wmb.WholeMemoryOptimizerType.OptAdaGrad
    elif str_wmb_optimizer == 'rmsprop':
        return wmb.WholeMemoryOptimizerType.OptRmsProp
    else:
        raise ValueError('WholeMemory optimizer %s not supported, should be (sgd, adam, adagrad, rmsprop)'
                         % (str_wmb_optimizer,))


def get_part_file_name(prefix: str,
                       part_id: int,
                       part_count: int):
    return '%s_part_%d_of_%d' % (prefix, part_id, part_count)


def get_part_file_list(prefix: str,
                       part_count: int):
    filelist = []
    for part_id in range(part_count):
        filelist.append('%s_part_%d_of_%d' % (prefix, part_id, part_count))
    return filelist