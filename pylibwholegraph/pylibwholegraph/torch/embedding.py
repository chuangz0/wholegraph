import pylibwholegraph.binding.wholememory_binding as wmb
import torch
import os
from .utils import torch_dtype_to_wholememory_dtype, get_file_size
from .utils import str_to_wmb_wholememory_location, str_to_wmb_wholememory_memory_type
from .utils import str_to_wmb_wholememory_optimizer_type, str_to_wmb_wholememory_access_type
from typing import Union, List
from .comm import WholeMemoryCommunicator
from .wholegraph_env import get_stream
from .tensor import WholeMemoryTensor
from .wholegraph_env import wrap_torch_tensor, get_wholegraph_env_fns, get_stream


class WholeMemoryOptimizer(object):
    def __init__(self, global_comm: WholeMemoryCommunicator):
        super().__init__()
        self.wmb_opt = None
        self.embeddings = []
        self.global_comm = None

    def add_embedding(self, wm_embedding):
        self.embeddings.append(wm_embedding)

    def step(self, lr: float):
        for wm_embedding in self.embeddings:
            if wm_embedding.need_apply:
                wm_embedding.apply_gradient(lr)
        self.global_comm.barrier()


def create_wholememory_optimizer(global_comm: WholeMemoryCommunicator, optimizer_type: str, param_dict: dict):
    wm_optimizer = WholeMemoryOptimizer(global_comm)
    wm_optimizer.wmb_opt.create_optimizer(str_to_wmb_wholememory_optimizer_type(optimizer_type), param_dict)


def destroy_wholememory_optimizer(optimizer: WholeMemoryOptimizer):
    optimizer.wmb_opt.destroy_optimizer()
    optimizer.wmb_opt = None


class WholeMemoryCachePolicy(object):
    def __init__(self):
        super().__init__()
        self.wmb_cache_policy = None


def create_wholememory_cache_policy(cache_comm: WholeMemoryCommunicator,
                                    *,
                                    memory_type: str = 'chunked',
                                    memory_location: str = 'cuda',
                                    access_type: str = 'readonly',
                                    ratio: float = 0.5):
    wmb_cache_policy = wmb.WholeMemoryCachePolicy()
    wmb_cache_policy.create_policy(cache_comm.wmb_comm,
                                   str_to_wmb_wholememory_memory_type(memory_type),
                                   str_to_wmb_wholememory_location(memory_location),
                                   str_to_wmb_wholememory_access_type(access_type),
                                   ratio)
    return wmb_cache_policy

def destroy_wholememory_cache_policy(wmb_cache_policy: WholeMemoryCachePolicy):
    wmb_cache_policy.destroy_policy()
    wmb_cache_policy.wmb_cache_policy = None


class EmbeddingLookupFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                indice: torch.Tensor,
                wm_embedding,
                force_dtype: Union[torch.dtype, None] = None):
        output_tensor = wm_embedding.gather(indice, force_dtype)
        if wm_embedding.need_grad():
            ctx.save_for_backward(indice, output_tensor)
            ctx.wm_embedding = wm_embedding
        return output_tensor

    @staticmethod
    def backward(ctx,
                 grad_outputs: torch.Tensor):
        indice, output_tensor = ctx.saved_tensors
        wm_embedding = ctx.wm_embedding
        wm_embedding.add_gradients(indice, wm_embedding)
        ctx.wm_embedding = None
        return None


class WholeMemoryEmbedding(object):
    r"""WholeMemory Embedding
    """
    def __init__(self,
                 wmb_embedding: wmb.PyWholeMemoryEmbedding,
                 wmb_optimizer: Union[WholeMemoryOptimizer, None],
                 wmb_cache_policy: Union[WholeMemoryCachePolicy, None]):
        super().__init__()
        self.wmb_embedding = wmb_embedding
        self.embedding_tensor = None
        self.optimizer_states = None

        self.wmb_optimizer = wmb_optimizer
        self.wmb_cache_policy = wmb_cache_policy

        self.adjust_cache = True if self.wmb_cache_policy is not None else False

        self.need_apply = False
        self.sparse_indices = []
        self.sparse_grads = []

    def set_adjust_cache(self, adjust_cache: bool):
        self.adjust_cache = adjust_cache if self.wmb_cache_policy is not None else False

    def need_grad(self):
        return self.wmb_optimizer is not None and torch.is_grad_enabled()

    def gather(self,
               indice: torch.Tensor,
               force_dtype: Union[torch.dtype, None] = None):
        assert indice.dim() == 1
        embedding_dim = self.get_embedding_tensor().shape[1]
        embedding_count = indice.shape[0]
        current_cuda_device = 'cuda:%d' % (torch.cuda.current_device(), )
        output_dtype = force_dtype if force_dtype is not None else self.embedding_tensor.dtype
        need_grad = self.need_grad()
        output_tensor = torch.empty([embedding_count, embedding_dim],
                                    device=current_cuda_device,
                                    dtype=output_dtype,
                                    requires_grad=need_grad)
        if need_grad:
            self.need_apply = True

    def add_gradients(self, indice: torch.Tensor, grad_outputs: torch.Tensor):
        self.sparse_indices.append(indice)
        self.sparse_grads.append(grad_outputs)

    def apply_gradients(self, lr: float):
        sparse_indices = torch.cat(self.sparse_indices)
        sparse_grads = torch.cat(self.sparse_grads)
        wmb.EmbeddingGatherGradientApply(self.wmb_embedding,
                                         wrap_torch_tensor(sparse_indices),
                                         wrap_torch_tensor(sparse_grads),
                                         self.adjust_cache,
                                         lr,
                                         get_wholegraph_env_fns(),
                                         get_stream())
        self.sparse_indices = []
        self.sparse_grads = []
        self.need_apply = []

    def writeback_all_cache(self):
        self.wmb_embedding.writeback_all_cache(get_stream(False))

    def drop_all_cache(self):
        self.wmb_embedding.drop_all_cache(get_stream(False))

    def get_embedding_tensor(self):
        if self.embedding_tensor is None:
            self.embedding_tensor = WholeMemoryTensor(self.wmb_embedding.get_embedding_tensor())
        return self.embedding_tensor

    def get_optimizer_state_names(self):
        return self.wmb_embedding.get_optimizer_state_names()

    def get_optimizer_state(self, state_name):
        if state_name not in self.optimizer_states:
            self.optimizer_states[state_name] = WholeMemoryTensor(self.wmb_embedding.get_optimizer_state(state_name))
        return self.optimizer_states[state_name]

    def save(self, file_prefix: str):
        self.get_embedding_tensor().to_file_prefix(file_prefix + '_embedding_tensor')
        for state_name in self.get_optimizer_state_names():
            state = self.get_optimizer_state(state_name)
            state.to_file_prefix(file_prefix + '_' + state_name)

    def load(self, file_prefix: str, *, ignore_embedding: bool = False, part_count: Union[int, None] = None):
        if ignore_embedding is False:
            self.get_embedding_tensor().from_file_prefix(file_prefix + '_embedding_tensor', part_count)
        for state_name in self.get_optimizer_state_names():
            state = self.get_optimizer_state(state_name)
            state.from_file_prefix(file_prefix + '_' + state_name, part_count)


def create_embedding(comm: WholeMemoryCommunicator,
                     memory_type: str,
                     memory_location: str,
                     dtype: torch.dtype,
                     sizes: List[int],
                     optimizer: Union[WholeMemoryOptimizer, None] = None,
                     cache_policy: Union[WholeMemoryCachePolicy, None] = None):
    r"""
    Create embedding
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param dtype: data type
    :param sizes: size of the embedding, must be 2D
    :param optimizer: optimizer
    :param cache_policy: cache policy
    :return: WholeMemoryEmbedding
    """
    if optimizer is None:
        wmb_optimizer = wmb.create_non_optimizer()
    else:
        wmb_optimizer = optimizer.wmb_opt
    if cache_policy is None:
        wmb_cache_policy = wmb.create_non_cache_policy()
    else:
        wmb_cache_policy = cache_policy.wmb_cache_policy
    assert len(sizes) == 2
    tensor_desc = wmb.PyWholeMemoryTensorDescription()
    tensor_desc.set_dtype(torch_dtype_to_wholememory_dtype(dtype))
    tensor_desc.set_shape(sizes)
    tensor_desc.set_stride([sizes[1], 1])
    wm_embedding = WholeMemoryEmbedding(
        wmb.create_embedding(tensor_desc,
                             comm.wmb_comm,
                             str_to_wmb_wholememory_memory_type(memory_type),
                             str_to_wmb_wholememory_location(memory_location),
                             wmb_optimizer,
                             wmb_cache_policy),
        optimizer,
        cache_policy)
    if optimizer is not None:
        optimizer.add_embedding(wm_embedding)
    return wm_embedding


def create_embedding_from_filelist(comm: WholeMemoryCommunicator,
                                   memory_type: str,
                                   memory_location: str,
                                   filelist: Union[List[str], str],
                                   dtype: torch.dtype,
                                   last_dim_size: int,
                                   optimizer: Union[WholeMemoryOptimizer, None] = None,
                                   cache_policy: Union[WholeMemoryCachePolicy, None] = None):
    r"""
    Create embedding from file list
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param filelist: list of files
    :param dtype: data type
    :param last_dim_size: size of last dim
    :param optimizer: optimizer
    :param cache_policy: cache policy
    :return:
    """
    if isinstance(filelist, str):
        filelist = [filelist]
    assert last_dim_size > 0
    element_size = torch.tensor([], dtype=dtype).element_size()
    file_entry_size = element_size * last_dim_size
    total_file_size = 0
    for filename in filelist:
        file_size = get_file_size(filename)
        if file_size % file_entry_size != 0:
            raise ValueError('File %s size is %d not mutlple of %d' % (filename, file_size, file_entry_size))
        total_file_size += file_size
    total_entry_count = total_file_size // file_entry_size
    wm_embedding = create_embedding(comm,
                                    memory_type,
                                    memory_location,
                                    dtype,
                                    [total_entry_count, last_dim_size],
                                    optimizer,
                                    cache_policy)
    wm_embedding.get_embedding_tensor().from_filelist(filelist)
    return wm_embedding


def destroy_embedding(wm_embedding: WholeMemoryEmbedding):
    wm_embedding.wmb_embedding.destroy_embedding()
    wm_embedding.wmb_embedding = None


class WholeMemoryEmbeddingModule(torch.nn.Module):
    def __init__(self, wm_embedding: WholeMemoryEmbedding):
        super().__init__()
        self.wm_embedding = wm_embedding
        self.embedding_gather_fn = EmbeddingLookupFn.apply

    def forward(self, indice: torch.Tensor, force_dtype: Union[torch.dtype, None] = None):
        return self.embedding_gather_fn(indice, self.wm_embedding, force_dtype)
