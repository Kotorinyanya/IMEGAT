from torch_geometric.data import Batch, Data
from torch_geometric.nn import dense_diff_pool, GATConv

# from torch_geometric.nn import GCNConv
from module import EGATConv, GraphConv
from torch_geometric.nn import GCNConv
from utils import *
from torch import nn
from torch_geometric.utils import to_scipy_sparse_matrix
from torch.nn import BatchNorm1d
from module import InstanceNorm, StablePool
from functools import partial
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, hiddem_dim, out_channels, depth,
                 first_conv_layer=GraphConv,
                 hidden_conv_layer=GraphConv,
                 last_conv_layer=GraphConv,
                 **kwargs):
        super(ResConvBlock, self).__init__()
        self.block_chunk_size = 2
        self.res_convs = nn.ModuleList()
        for i in range(depth):
            if i == 0:  # first layer
                self.res_convs.append(first_conv_layer(in_channels, hiddem_dim))
                self.res_convs.append(InstanceNorm(hiddem_dim))
            elif i == depth - 1:  # last layer
                self.res_convs.append(hidden_conv_layer(hiddem_dim * 2, out_channels))
                self.res_convs.append(InstanceNorm(out_channels))
            else:  # hidden layer
                self.res_convs.append(last_conv_layer(hiddem_dim * 2, hiddem_dim))
                self.res_convs.append(InstanceNorm(hiddem_dim))

    def forward(self, x, edge_index, edge_attr, batch_mask):
        out_all = []

        if self.block_chunk_size == 2:  # with batch norm
            for i, (conv_block, norm_block) in enumerate(chunks(self.res_convs, self.block_chunk_size)):
                # print(i, "ea", ea[:10].view(-1))
                if i >= 1:  # add res block
                    x = torch.cat([out_all[-1], x], dim=-1)
                x = conv_block(x, edge_index, edge_attr.view(-1))
                x = F.leaky_relu(x, negative_slope=0.2)
                # print(i, "x", x[:5, :5])
                x = norm_block(x, batch_mask)
                out_all.append(x)

        return out_all

    @property
    def device(self):
        return self.res_convs[0].weight.device


class ParallelResGraphConv(nn.Module):
    """
    handel multi-dimensional edge_attr
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dims=1, depth=3):
        super(ParallelResGraphConv, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.dims = dims
        self.parallel_convs = nn.ModuleList([
            ResConvBlock(in_channels, hidden_channels, out_channels, self.depth)
            for _ in range(self.dims)
        ])

    def forward(self, x, edge_index, edge_attr, batch_mask):
        assert self.dims == edge_attr.shape[1]  # multi-dimensional edge_attr

        return [self.parallel_convs[i](x, edge_index, edge_attr[:, i].view(-1), batch_mask) for i in range(self.dims)]

    @property
    def device(self):
        return self.parallel_convs[0].device

    def __repr__(self):
        string = '{}({}, {}, {}, dims={}, depth={})'.format(
            self.__class__.__name__,
            self.in_channels, self.hidden_channels, self.out_channels,
            self.dims, self.depth)
        return string
