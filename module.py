import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_sort_pool
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.inits import zeros, uniform, glorot
from torch_geometric.data import Batch
from boxx import timeit

import torch.nn.functional as F
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops, dropout_adj
from torch_scatter import scatter_add

from utils import add_self_loops_with_edge_attr, real_softmax, nan_or_inf


class MEGATConv(torch.nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 out_channels,
                 att_dropout=0,
                 bias=True,
                 concat=True):
        super(MEGATConv, self).__init__()

        self.convs = nn.ModuleList([
            EGATConv(in_channels, out_channels, att_dropout=att_dropout, bias=bias)
            for _ in range(heads)
        ])

    def forward(self, x, edge_index, edge_attr):
        pass


class EGATConv(torch.nn.Module):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): Whether to concat or average multi-head
            attentions (default: :obj:`True`)
        att_dropout (float, optional): Dropout probability of the normalized
            attention coefficients, i.e. exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 att_dropout=0,
                 bias=True):
        super(EGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_dropout = att_dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.att_weight = Parameter(torch.Tensor(2 * in_channels, 1))

        self.att_drop = nn.Dropout(att_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        self.weight.data = nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        self.att_weight.data = nn.init.xavier_uniform_(self.att_weight.data, gain=nn.init.calculate_gain('relu'))
        # uniform(self.att_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        row, col = edge_index

        # Compute attention coefficients
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = (alpha @ self.att_weight).squeeze(-1)
        # This will broadcast edge_attr across all attentions
        alpha = alpha * edge_attr.abs()
        alpha = F.leaky_relu(alpha, negative_slope=10)
        alpha = softmax(alpha, row)
        # Dropout attentions
        edge_index, alpha = dropout_adj(edge_index, alpha, self.att_dropout)

        x = x @ self.weight

        edge_index, alpha = add_self_loops(edge_index, alpha)
        row, col = edge_index

        # Sum up neighborhoods.
        out = alpha.view(-1, 1) * x[col]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.bias is not None:
            out = out + self.bias

        assert not nan_or_inf(out)

        return out, alpha, edge_index, edge_attr

    def __repr__(self):
        return '{}({}, {}, att_dropout={})'.format(self.__class__.__name__, self.in_channels,
                                                   self.out_channels, self.att_dropout)


if __name__ == '__main__':
    from utils import z_score_norm_data
    from dataset import ABIDE

    dataset = ABIDE(root='datasets/NYU', transform=z_score_norm_data)
    conv = EGATConv(11, 30)
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([data])
    conv(batch.x, batch.edge_index, batch.edge_attr)
    pass
