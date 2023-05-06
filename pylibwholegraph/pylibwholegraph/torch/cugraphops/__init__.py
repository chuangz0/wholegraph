import warnings
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor

from pylibcugraphops.pytorch import (
    SampledCSC,
    SampledHeteroCSC,
    StaticCSC,
    StaticHeteroCSC,
)


class CuGraphModule(torch.nn.Module):  # pragma: no cover
    r"""An abstract base class for implementing cugraph message passing layers."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def get_cugraph(
        self,
        csc: Tuple[Tensor, Tensor, int],
    ) -> Any:
        r"""Constructs a :obj:`cugraph` graph object from CSC representation.
        Supports both bipartite and non-bipartite graphs.
        Args:
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`CuGraphModule.to_csc` method to convert an
                :obj:`edge_index` representation to the desired format.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, will be computed on-the-fly,
                leading to slightly worse performance. (default: :obj:`None`)
        """
        row, colptr, num_src_nodes = csc

        if not row.is_cuda:
            raise RuntimeError(
                f"'{self.__class__.__name__}' requires GPU-"
                f"based processing (got CPU tensor)"
            )

        return SampledCSC(colptr, row, max_num_neighbors, num_src_nodes)

    def get_typed_cugraph(
        self,
        csc: Tuple[Tensor, Tensor, int],
        edge_type: Tensor,
        num_edge_types: Optional[int] = None,
        max_num_neighbors: Optional[int] = None,
    ) -> Any:
        r"""Constructs a typed :obj:`cugraph` graph object from a CSC
        representation where each edge corresponds to a given edge type.
        Supports both bipartite and non-bipartite graphs.
        Args:
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`CuGraphModule.to_csc` method to convert an
                :obj:`edge_index` representation to the desired format.
            edge_type (torch.Tensor): The edge type.
            num_edge_types (int, optional): The maximum number of edge types.
                When not given, will be computed on-the-fly, leading to
                slightly worse performance. (default: :obj:`None`)
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, will be computed on-the-fly,
                leading to slightly worse performance. (default: :obj:`None`)
        """
        if num_edge_types is None:
            num_edge_types = int(edge_type.max()) + 1

        row, colptr, num_src_nodes = csc
        edge_type = edge_type.int()

        if num_src_nodes != colptr.numel() - 1:  # Bipartite graph:
            if max_num_neighbors is None:
                max_num_neighbors = int((colptr[1:] - colptr[:-1]).max())

            if LEGACY_MODE:
                dst_nodes = torch.arange(colptr.numel() - 1, device=row.device)
                return make_mfg_csr_hg(
                    dst_nodes,
                    colptr,
                    row,
                    max_num_neighbors,
                    num_src_nodes,
                    n_node_types=0,
                    n_edge_types=num_edge_types,
                    out_node_types=None,
                    in_node_types=None,
                    edge_types=edge_type,
                )

            return SampledHeteroCSC(
                colptr, row, edge_type, max_num_neighbors, num_src_nodes, num_edge_types
            )

        if LEGACY_MODE:
            return make_fg_csr_hg(
                colptr,
                row,
                n_node_types=0,
                n_edge_types=num_edge_types,
                node_types=None,
                edge_types=edge_type,
            )

        return StaticHeteroCSC(colptr, row, edge_type, num_edge_types)

    def forward(
        self,
        x: Tensor,
        csc: Tuple[Tensor, Tensor, int],
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.
        Args:
            x (torch.Tensor): The node features.
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`CuGraphModule.to_csc` method to convert an
                :obj:`edge_index` representation to the desired format.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, the value will be computed
                on-the-fly, leading to slightly worse performance.
                (default: :obj:`None`)
        """
        raise NotImplementedError
