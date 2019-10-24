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
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_scatter import scatter_add

from utils import add_self_loops_with_edge_attr, real_softmax


class EGATConv(torch.nn.Module):
    """
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): Whether to concat or average multi-head
            attentions (default: :obj:`True`)
        negative_slope (float, optional): LeakyRELU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients, i.e. exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0,
                 bias=True):
        super(EGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.drop = nn.Dropout(dropout)

        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.att_weight = Parameter(torch.Tensor(1, 2 * out_channels))

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

        x = torch.mm(x.float(), self.weight)

        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        row, col = edge_index

        # Compute attention coefficients
        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = (alpha * self.att_weight).sum(dim=-1)
        # This will broadcast edge_attr across all attentions
        alpha = torch.mul(alpha, edge_attr.abs())
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # alpha = F.relu(alpha)
        # alpha = F.normalize(alpha, p=1, dim=1)
        alpha = real_softmax(alpha, row)
        # alpha = F.sigmoid(alpha)

        # Sum up neighborhoods.
        out = alpha.view(-1, 1) * x[col]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.bias is not None:
            out = out + self.bias

        assert torch.isnan(out).sum() == 0

        return out, alpha

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Pool(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_blocks=3):
        super(Pool, self).__init__()
        self.conv_blocks = conv_blocks

    def forward(self, x, edge_index, edge_attr):
        pass

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    from utils import gaussian_fit
    from dataset import ABIDE

    dataset = ABIDE(root='datasets/NYU', transform=gaussian_fit)
    conv = EGATConv(7, 30)
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([data])
    conv(batch)
    pass
