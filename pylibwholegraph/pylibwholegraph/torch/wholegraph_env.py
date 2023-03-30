import torch
import pylibwholegraph.binding.wholememory_binding as wmb
from enum import IntEnum
import sys
from typing import Union


default_cuda_stream_int_ptr = None
default_wholegraph_env_context = None


def get_stream(use_default = True):
    global default_cuda_stream_int_ptr
    cuda_stream_int_ptr = None
    if default_cuda_stream_int_ptr is None or use_default == False:
        cuda_stream = torch.cuda.current_stream()._as_parameter_
        if cuda_stream.value is not None:
            cuda_stream_int_ptr = cuda_stream.value
        else:
            cuda_stream_int_ptr = int(0)
        if use_default:
            default_cuda_stream_int_ptr = cuda_stream_int_ptr
    else:
        cuda_stream_int_ptr = default_cuda_stream_int_ptr
    return cuda_stream_int_ptr


def wholememory_dtype_to_pytorch_dtype(wm_dtype: wmb.WholeMemoryDataType):
    if wm_dtype == wmb.WholeMemoryDataType.DtFloat:
        return torch.float
    elif wm_dtype == wmb.WholeMemoryDataType.DtHalf:
        return torch.half
    elif wm_dtype == wmb.WholeMemoryDataType.DtBF16:
        return torch.bfloat16
    elif wm_dtype == wmb.WholeMemoryDataType.DtInt:
        return torch.int
    elif wm_dtype == wmb.WholeMemoryDataType.DtInt64:
        return torch.int64
    elif wm_dtype == wmb.WholeMemoryDataType.DtInt8:
        return torch.int8
    elif wm_dtype == wmb.WholeMemoryDataType.DtDouble:
        return torch.double
    elif wm_dtype == wmb.WholeMemoryDataType.DtInt16:
        return torch.int16
    else:
        raise ValueError('Invalid wmb.WholeMemoryDataType')


def pytorch_dtype_to_wholememory_dtype(th_dtype: torch.dtype):
    if th_dtype == torch.float:
        return wmb.WholeMemoryDataType.DtFloat
    elif th_dtype == torch.half:
        return wmb.WholeMemoryDataType.DtHalf
    elif th_dtype == torch.bfloat16:
        return wmb.WholeMemoryDataType.DtBF16
    elif th_dtype == torch.int:
        return wmb.WholeMemoryDataType.DtInt
    elif th_dtype == torch.int64:
        return wmb.WholeMemoryDataType.DtInt64
    elif th_dtype == torch.int8:
        return wmb.WholeMemoryDataType.DtInt8
    elif th_dtype == torch.double:
        return wmb.WholeMemoryDataType.DtDouble
    elif th_dtype == torch.int16:
        return wmb.WholeMemoryDataType.DtInt16
    else:
        raise ValueError('Not supported torch.dtype')


class TorchEmptyGlobalContext(object):
    def __init__(self):
        pass


class TorchMemoryContext(object):
    def __init__(self):
        self.tensor = None

    def set_tensor(self, t: torch.Tensor):
        self.tensor = t

    def get_tensor(self):
        return self.tensor

    def free(self):
        self.tensor = None


def torch_create_memory_context_env_fn(global_context: TorchEmptyGlobalContext) -> TorchMemoryContext:
    t = TorchMemoryContext()
    #print('torch_create_memory_context_env_fn t=%d' % (id(t), ))
    return t


def torch_destroy_memory_context_env_fn(memory_context: TorchMemoryContext,
                                        global_context: TorchEmptyGlobalContext):
    pass


def torch_malloc_env_fn(tensor_desc: wmb.PyWholeMemoryTensorDescription,
                        malloc_type: wmb.PyMemoryAllocType,
                        memory_context: TorchMemoryContext,
                        global_context: TorchEmptyGlobalContext) -> int:
    #print('already in torch_malloc_env_fn', file=sys.stderr)
    pinned = False
    device = None
    #print('torch_malloc_env_fn before config, type=%d' % (malloc_type.get_type(), ), file=sys.stderr)
    if malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatDevice:
        device = torch.device('cuda')
    elif malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatHost:
        device = torch.device('cpu')
    else:
        assert malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatPinned
        device = torch.device('cpu')
        pinned = True
    #print('torch_malloc_env_fn after config', file=sys.stderr)
    shape = tensor_desc.shape
    #print('torch_malloc_env_fn after shape', file=sys.stderr)
    dtype = wholememory_dtype_to_pytorch_dtype(tensor_desc.dtype)
    #print('torch_malloc_env_fn after dtype', file=sys.stderr)
    t = torch.empty(shape,
                    dtype=dtype,
                    device=device,
                    pin_memory=pinned)
    memory_context.set_tensor(t)
    #print('torch_malloc_env_fn done return=%ld' % (t.data_ptr(), ), file=sys.stderr)
    return t.data_ptr()


def torch_free_env_fn(memory_context: TorchMemoryContext,
                      global_context: TorchEmptyGlobalContext):
    memory_context.free()


def create_current_env_context():
    #print('in wholegraph_env.py create_current_env_context')
    context = wmb.GlobalContextWrapper()
    global_context = TorchEmptyGlobalContext()
    context.create_context(torch_create_memory_context_env_fn,
                           torch_destroy_memory_context_env_fn,
                           torch_malloc_env_fn,
                           torch_free_env_fn,
                           global_context,
                           torch_malloc_env_fn,
                           torch_free_env_fn,
                           global_context)
    return context


def get_wholegraph_env_fns(use_default = True) -> int:
    global default_wholegraph_env_context
    wholegraph_env_context = None
    if default_wholegraph_env_context is None or use_default == False:
        wholegraph_env_context = create_current_env_context()
        if use_default:
            default_wholegraph_env_context = wholegraph_env_context
    else:
        wholegraph_env_context = default_wholegraph_env_context
    return wholegraph_env_context.get_env_fns()


def wrap_torch_tensor(t: Union[torch.Tensor, None]) -> wmb.WrappedLocalTensor:
    py_desc = wmb.PyWholeMemoryTensorDescription()
    wm_t = wmb.WrappedLocalTensor()
    if t is None:
        return wm_t.wrap_tensor(py_desc, 0)
    py_desc.set_dtype(pytorch_dtype_to_wholememory_dtype(t.dtype))
    py_desc.set_storage_offset(0)
    py_desc.set_shape(tuple(t.shape))
    py_desc.set_stride(tuple(t.stride()))
    return wm_t.wrap_tensor(py_desc, t.data_ptr())

