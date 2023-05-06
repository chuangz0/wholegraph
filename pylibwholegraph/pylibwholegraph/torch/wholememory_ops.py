import torch
import pylibwholegraph.binding.wholememory_binding as wmb
from .wholegraph_env import (
    get_stream,
    get_wholegraph_env_fns,
    wrap_torch_tensor,
)
from .utils import wholememory_dtype_to_torch_dtype


def wholememory_gather_forward_functor(
    wholememory_tensor: wmb.PyWholeMemoryTensor,
    indices_tensor: torch.Tensor,
    requires_grad=False,
    torch_output_dtype=None,
):
    assert indices_tensor.dim() == 1
    assert indices_tensor.dtype == torch.int32 or indices_tensor.dtype == torch.int64
    if torch_output_dtype is None:
        torch_output_dtype = wholememory_dtype_to_torch_dtype(wholememory_tensor.dtype)
    output_tensor = torch.empty(
        [indices_tensor.shape[0], wholememory_tensor.shape[1]],
        device="cuda",
        dtype=torch_output_dtype,
        requires_grad=requires_grad,
    )
    wmb.wholememory_gather_op(
        wholememory_tensor,
        wrap_torch_tensor(indices_tensor),
        wrap_torch_tensor(output_tensor),
        get_wholegraph_env_fns(),
        get_stream(),
    )
    return output_tensor


def wholememory_scatter_functor(
    input_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    wholememory_tensor: wmb.PyWholeMemoryTensor,
):
    assert indices_tensor.dim() == 1
    assert indices_tensor.dtype == torch.int32 or indices_tensor.dtype == torch.int64
    wmb.wholememory_scatter_op(
        wrap_torch_tensor(input_tensor),
        wrap_torch_tensor(indices_tensor),
        wholememory_tensor,
        get_wholegraph_env_fns(),
        get_stream(),
    )
