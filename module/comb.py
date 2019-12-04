import torch.nn.functional as F
from torch import nn

from module import EGATConv, GraphConv
from module import InstanceNorm
from utils import *


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, depth,
                 first_conv_layer=GraphConv,
                 hidden_conv_layer=GraphConv,
                 last_conv_layer=GraphConv,
                 **kwargs):
        super(ResConvBlock, self).__init__()
        self.block_chunk_size = 2
        self.res_convs = nn.ModuleList()
        for i in range(depth):
            if i == 0:  # first layer
                self.res_convs.append(first_conv_layer(in_channels, hidden_dim))
                self.res_convs.append(InstanceNorm(hidden_dim))
            elif i == depth - 1:  # last layer
                self.res_convs.append(hidden_conv_layer(hidden_dim * 2, out_channels))
                self.res_convs.append(InstanceNorm(out_channels))
            else:  # hidden layer
                self.res_convs.append(last_conv_layer(hidden_dim * 2, hidden_dim))
                self.res_convs.append(InstanceNorm(hidden_dim))

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
        x = x.reshape(x.shape[0], -1, self.dims)

        return [self.parallel_convs[i](x[:, :, i], edge_index, edge_attr[:, i].view(-1), batch_mask)
                for i in range(self.dims)]

    @property
    def device(self):
        return self.parallel_convs[0].device

    def __repr__(self):
        string = '{}({}, {}, {}, dims={}, depth={})'.format(
            self.__class__.__name__,
            self.in_channels, self.hidden_channels, self.out_channels,
            self.dims, self.depth)
        return string


class ParallelEGAT(nn.Module):
    """
    handel multi-dimensional edge_attr
    """

    def __init__(self, in_channels, out_channels, dims, att_dropout=0.):
        super(ParallelEGAT, self).__init__()
        self.att_dropout = att_dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims
        self.parallel_convs = nn.ModuleList([
            EGATConv(in_channels, out_channels, heads=1, concat=False, att_dropout=self.att_dropout)
            for _ in range(self.dims)
        ])

    def forward(self, x, edge_index, edge_attr):
        assert self.dims == edge_attr.shape[1]  # multi-dimensional edge_attr
        x = x.reshape(x.shape[0], -1, self.dims)

        out_l, alpha_l, alpha_index_l = [], [], []
        for i in range(self.dims):
            out_i, alpha_i, alpha_index_i = self.parallel_convs[i](x[:, :, i], edge_index, edge_attr[:, i].view(-1))
            out_l.append(out_i)
            alpha_l.append(alpha_i)
            alpha_index_l.append(alpha_index_i)
        out = torch.cat(out_l, dim=-1)
        alpha = torch.cat(alpha_l, dim=-1)
        alpha_index = alpha_index_l[0]
        return out, alpha, alpha_index

    @property
    def device(self):
        return self.parallel_convs[0].device

    def __repr__(self):
        string = '{}({}, {}, {}, dims={}, depth={})'.format(
            self.__class__.__name__,
            self.in_channels, self.hidden_channels, self.out_channels,
            self.dims, self.depth)
        return string


from .pooling import Pool


class ConvNPool(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_dim,
                 out_channels,
                 attention_heads,
                 in_dims,
                 concat,
                 att_dropout,
                 conv_depth,
                 pool_conv_depth,
                 ml=0,
                 el=0,
                 ll=0,
                 pool_nodes=None,
                 no_pool=False):
        """

        :param in_channels:
        :param hidden_dim:
        :param out_channels:
        :param attention_heads:
        :param in_dims: multi-dimensional graph
        :param concat: for EGAT
        :param att_dropout: attention dropout
        :param conv_depth: ResConv
        :param pool_conv_depth: ResConv in Pool
        :param pool_nodes:
        :param ml: c for modularity loss
        :param el: c for entropy loss, large c for a hard pooling, small c for soft pooling
        """
        super(ConvNPool, self).__init__()
        self.ll = ll
        self.no_pool = no_pool
        self.el = el
        self.ml = ml
        self.in_dims = in_dims
        self.pool_nodes = pool_nodes
        self.out_channels = out_channels
        self.pool_depth = pool_conv_depth
        self.conv_depth = conv_depth
        self.att_dropout = att_dropout
        self.concat = concat
        self.attention_heads = attention_heads
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels

        # EGAT
        if self.in_dims == 1:
            self.attention_dim = self.attention_heads if self.concat else 1
            self.egat_conv = EGATConv(
                self.in_channels, self.hidden_dim, heads=self.attention_heads, concat=self.concat, att_dropout=0.)
        elif self.in_dims > 1 and self.attention_heads == 1:
            self.attention_dim = self.in_dims
            self.egat_conv = ParallelEGAT(
                self.in_channels, self.hidden_dim, dims=self.in_dims, att_dropout=0.)
        else:
            raise Exception("???")
        self.bn = InstanceNorm(hidden_dim)

        # ResConv
        self.conv = ParallelResGraphConv(
            self.hidden_dim, self.hidden_dim, self.out_channels, self.attention_dim, self.conv_depth)

        # Pool
        if not self.no_pool:
            self.pool = Pool(
                self.in_channels, self.hidden_dim, self.pool_nodes, self.pool_depth, self.attention_dim, self.ml,
                self.el, self.ll)

    def forward(self, x, edge_index, edge_attr, batch):

        # attention
        x1, alpha, alpha_index = self.egat_conv(x, edge_index, edge_attr)
        self.alpha, self.alpha_index = alpha, alpha_index

        # conv
        conv_out = self.conv(x1, alpha_index, alpha, batch.batch.to(self.device))
        out_all = torch.cat([torch.cat(d, dim=1) for d in conv_out], dim=-1)
        out_last = torch.cat([d[-1] for d in conv_out], dim=-1)

        if not self.no_pool:
            # pool
            x1_to_pool = out_last
            x_in = torch.stack([x for _ in range(self.attention_dim)], dim=-1) if x.dim() == 2 else x
            p1_x, p1_ei, p1_ea, p1_batch, p1_loss, assignment = self.pool(x_in, alpha_index, alpha, x1_to_pool, batch)
            return out_all, p1_x, p1_ei, p1_ea, p1_batch, p1_loss, assignment
        else:
            # without pool
            return out_last

    @property
    def device(self):
        return self.egat_conv.device

    def __repr__(self):
        return '{}({}, {}, {}, pool_nodes={} attention_dim={}, concat={}, att_dropout={}, ' \
               'conv_depth={}, pool_conv_depth={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_dim, self.out_channels, self.pool_nodes,
            self.attention_dim,
            self.concat, self.att_dropout, self.conv_depth, self.pool_depth)
