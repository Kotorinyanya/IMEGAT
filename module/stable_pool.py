import torch
from torch import nn


class StablePool(nn.Module):
    def __init__(self, in_nodes, out_nodes, beta):
        super(StablePool, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.beta = beta
        self.s2 = nn.Parameter(torch.Tensor(self.in_nodes, self.out_nodes))

        self.reset_parameters()

    def reset_parameters(self):
        self.s2.data = nn.init.xavier_uniform_(self.s2.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, s1):
        """

        :param s1: Tensor with dim (num_graphs, num_nodes, x)
        :return:
        """
        assert s1.dim() == 3
        return (1 / self.beta) * s1 + (self.beta - 1 / self.beta) * self.s2

    def __repr__(self):
        return '{}({}, {}, beta={})'.format(self.__class__.__name__, self.in_nodes,
                                            self.out_nodes, self.beta)
