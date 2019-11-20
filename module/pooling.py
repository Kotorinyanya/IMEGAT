from torch_geometric.data import Batch, Data
from torch_geometric.nn import dense_diff_pool, GATConv

# from torch_geometric.nn import GCNConv
from module import EGATConv, GraphConv
from torch_geometric.nn import GCNConv
from utils import *
from torch import nn
from torch_geometric.utils import to_scipy_sparse_matrix
from torch.nn import BatchNorm1d
from module import ParallelResGraphConv
from functools import partial
import torch.nn.functional as F

EPS = 1e-15


class Pool(nn.Module):
    def __init__(self, in_channels, hidden_dim, pool_nodes, depth, dims):
        super(Pool, self).__init__()
        self.dims = dims
        self.block_chunk_size = 2
        self.pool_nodes = pool_nodes
        self.detach_pool = True  # detach adj
        self.pooling_on_adj = False
        self.conv_depth = depth
        self.pool_convs = ParallelResGraphConv(in_channels, hidden_dim, pool_nodes, dims=self.dims,
                                               depth=self.conv_depth)
        self.pool_fc = nn.Sequential(
            nn.Linear((hidden_dim * (self.conv_depth - 1) + pool_nodes) * self.dims, 50),
            nn.Linear(50, pool_nodes)
        )

        self.out_x_fc = nn.Sequential(
            nn.Linear(hidden_dim * self.dims, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, batch, x_to_pool):
        # reduce dimension first
        x_to_pool = self.out_x_fc(x_to_pool)

        # variables
        batch_mask = batch.batch.to(self.device)
        num_graphs = batch.num_graphs
        num_nodes = int(batch.num_nodes / batch.num_graphs)
        self.edge_index, self.edge_attr = edge_index, edge_attr

        # convert edge_index to adj
        adj = batch_to_adj(edge_index, edge_attr, num_nodes, num_graphs)

        # ParallelResGraphConv
        pool_conv_out_all = self.pool_convs(x, edge_index, edge_attr, batch_mask)
        pool_conv_out_all = torch.cat([torch.cat(d, dim=1) for d in pool_conv_out_all], dim=-1)
        assignment = self.pool_fc(pool_conv_out_all)
        # softmax
        self.pool_assignment = self.split_n(assignment, num_graphs)
        self.pool_assignment = torch.softmax(self.pool_assignment, dim=-1)

        # loss for S
        loss = self.losses()

        # perform pooling
        # self.pool_assignment = self.pool_assignment.detach() if self.detach_pool else self.pool_assignment
        pooled_x = self.pool_assignment.transpose(1, 2) @ self.split_n(x_to_pool, num_graphs)
        pooled_adj = self.pool_assignment.transpose(1, 2) @ adj @ self.pool_assignment

        pooled_adj = pooled_adj.permute(1, 0, 2, 3)
        # convert adj to edge_index
        data_list = []
        for i in range(num_graphs):
            tmp_adj = pooled_adj[i]
            # tmp_adj /= (adj.shape[-1] / pooled_adj.shape[-1]) ** 2  # normalize?
            if self.pooling_on_adj:
                tmp_adj = torch.max(tmp_adj, 0)[0].unsqueeze(0)  # max
                # tmp_adj = torch.mean(tmp_adj, 0).unsqueeze(0)  # mean
            edge_index, edge_attr = from_3d_tensor_adj(tmp_adj.detach() if self.detach_pool else tmp_adj.clone())
            tmp_data = Data(x=pooled_x[i], edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.pool_nodes)
            data_list.append(tmp_data)
        pooled_batch = Batch.from_data_list(data_list)
        # pooled_batch.to_data_list()
        pooled_edge_index, pooled_edge_attr = pooled_batch.edge_index, pooled_batch.edge_attr
        pooled_x = pooled_x.reshape(-1, pooled_x.shape[-1])  # merge to batch
        # pooled_x /= (adj.shape[-1] / pooled_adj.shape[-1])  # normalize?

        return pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, loss

    def losses(self):
        modularity_loss = sum([
            self.modularity_loss(self.pool_assignment.view(-1, self.pool_nodes), self.edge_index, self.edge_attr[:, i])
            for i in range(self.dims)],
            0)
        entropy_loss = self.entropy_loss(self.pool_assignment)
        # link_loss = self.link_loss(self.pool_assignment, adj)
        return modularity_loss + entropy_loss

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])

    @staticmethod
    def modularity_loss(assignment, edge_index, edge_attr=None):
        assert edge_attr.dim() == 1
        edge_attr = 1 if edge_attr is None else edge_attr
        row, col = edge_index
        reg = (edge_attr * torch.pow((assignment[row] - assignment[col]), 2).sum(1)).mean()
        return reg

    @staticmethod
    def entropy_loss(assignment):
        return (-assignment * torch.log(assignment + EPS)).sum(dim=-1).mean()

    @staticmethod
    def link_loss(assignment, adj):
        return torch.norm(adj - torch.matmul(assignment, assignment.transpose(1, 2)), p=2) / adj.numel()

    @property
    def device(self):
        return self.pool_convs.device
