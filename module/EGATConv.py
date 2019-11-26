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

from utils import add_self_loops_with_edge_attr, real_softmax, nan_or_inf, add_self_loops_mul


class Attention(nn.Module):
    def __init__(self,
                 channels,
                 heads=1,
                 concat=True,
                 att_dropout=0):
        super(Attention, self).__init__()

        self.concat = concat
        self.att_dropout = att_dropout
        self.heads = heads
        self.channels = channels
        self.att_drop = nn.Dropout(att_dropout)
        self.alpha_fc = nn.Sequential(
            nn.Linear(2 * channels, self.heads),
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        row, col = edge_index

        # Compute attention coefficients
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = self.alpha_fc(alpha)
        # This will broadcast edge_attr across all attentions

        # sigmoid
        # alpha = torch.sigmoid(alpha) * edge_attr.abs().reshape(-1, 1)

        # softmax
        alpha = alpha * edge_attr.abs().reshape(-1, 1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha *= 100  # de-flatten
        alpha = softmax(alpha, row)

        assert not nan_or_inf(alpha)

        # Dropout attentions
        if self.att_dropout > 0:
            edge_index, alpha = dropout_adj(edge_index, alpha, self.att_dropout, training=self.training)
        # # Add self-loop to alpha
        # edge_index, alpha = add_self_loops_mul(edge_index, alpha)

        if not self.concat:
            alpha = alpha.mean(-1).reshape(-1, 1)

        return alpha, edge_index

    @property
    def device(self):
        return self.att_weight.device

    def __repr__(self):
        return '{}({}, heads={}, concat={}, att_dropout={})'.format(self.__class__.__name__, self.channels,
                                                                    self.heads, self.concat, self.att_dropout)


class EGATConv(nn.Module):
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
                 heads=1,
                 concat=False,
                 att_dropout=0,
                 bias=False):
        super(EGATConv, self).__init__()

        self.concat = concat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_dropout = att_dropout
        self.heads = heads

        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.attention = Attention(out_channels, heads, concat, att_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels)) if not self.concat else \
                Parameter(torch.Tensor(out_channels * self.heads))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data = nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        x = x @ self.weight

        alpha, alpha_index = self.attention(x, edge_index, edge_attr)

        row, col = alpha_index
        out = self.my_cast(alpha, x[col])
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.bias is not None:
            out = out + self.bias

        assert not nan_or_inf(out)
        return out, alpha, alpha_index

    @staticmethod
    def my_cast(alpha, x):
        results = []
        for i in range(alpha.shape[1]):
            result = alpha[:, i].reshape(-1, 1) * x
            results.append(result)
        return torch.stack(results, dim=-1)

    @property
    def device(self):
        return self.weight.device

    def __repr__(self):
        return '{}({}, {}, heads={}, concat={}, att_dropout={})'.format(self.__class__.__name__, self.in_channels,
                                                                        self.out_channels, self.heads, self.concat,
                                                                        self.att_dropout)


if __name__ == '__main__':
    from utils import z_score_norm_data
    from dataset import ABIDE

    dataset = ABIDE(root='../datasets/NYU', transform=z_score_norm_data)
    conv = EGATConv(7, 30, heads=5, concat=True)
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([data])
    conv(batch.x, batch.edge_index, batch.edge_attr)
    pass
