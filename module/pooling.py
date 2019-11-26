from torch import nn
from torch_geometric.data import Batch, Data

from .comb import ParallelResGraphConv
from utils import *

EPS = 1e-15


class Pool(nn.Module):
    def __init__(self, in_channels, hidden_dim, pool_nodes, depth, dims):
        super(Pool, self).__init__()
        self.dims = dims
        self.block_chunk_size = 2
        self.pool_nodes = pool_nodes
        self.detach_pool = True  # detach adj
        self.conv_depth = depth
        self.pool_convs = ParallelResGraphConv(in_channels, hidden_dim, pool_nodes, dims=self.dims,
                                               depth=self.conv_depth)
        self.pool_fc = nn.Sequential(
            nn.Linear((hidden_dim * (self.conv_depth - 1) + pool_nodes), 50),
            nn.Linear(50, pool_nodes)
        )
        self.pool_fc = nn.ModuleList([self.pool_fc for _ in range(self.dims)])

        # self.out_x_fc = nn.Sequential(
        #     nn.Linear(hidden_dim * self.dims, hidden_dim),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )

    def forward(self, x, edge_index, edge_attr, x_to_pool, batch):
        # variables
        batch_mask = batch.batch.to(self.device)
        num_graphs = batch.num_graphs
        num_nodes = int(batch.num_nodes / batch.num_graphs)
        self.edge_index, self.edge_attr = edge_index, edge_attr

        # convert edge_index to adj
        adj = batch_to_adj(edge_index, edge_attr, num_nodes, num_graphs)

        # ParallelResGraphConv
        pool_conv_out_all = self.pool_convs(x, edge_index, edge_attr, batch_mask)

        # pool on every dims separately
        assignments = []
        for i in range(self.dims):
            conv_out = pool_conv_out_all[i]
            conv_out = torch.cat(conv_out, dim=-1)
            assignment_i = self.pool_fc[i](conv_out)
            assignments.append(assignment_i)
        assignment = torch.stack(assignments, dim=0)
        # pool_conv_out_all = torch.cat([torch.cat(d, dim=1) for d in pool_conv_out_all], dim=-1)
        # assignment = self.pool_fc(pool_conv_out_all)
        # softmax
        self.pool_assignment = assignment.reshape(self.dims, num_graphs, num_nodes, self.pool_nodes)
        self.pool_assignment = torch.softmax(self.pool_assignment, dim=-1)

        # loss for S
        loss = self.losses()

        # perform pooling
        # self.pool_assignment = self.pool_assignment.detach() if self.detach_pool else self.pool_assignment
        x_to_pool = x_to_pool.reshape(x_to_pool.shape[0], -1, self.dims)
        x_to_pool = x_to_pool.permute(2, 0, 1).reshape(self.dims, num_graphs, num_nodes, -1)

        pooled_x = self.pool_assignment.transpose(-2, -1) @ x_to_pool
        pooled_adj = self.pool_assignment.transpose(-2, -1) @ adj @ self.pool_assignment

        pooled_x = pooled_x.permute(1, 2, 3, 0)
        pooled_x = pooled_x.reshape(-1, pooled_x.shape[-2], pooled_x.shape[-1])  # merge to batch
        pooled_x /= (num_nodes / self.pool_nodes)  # normalize?
        pooled_adj /= (num_nodes / self.pool_nodes) ** 2  # normalize?

        # convert adj to edge_index
        # pooled_edge_index, pooled_edge_attr, pooled_batch_mask = adj_to_batch(pooled_adj)
        pooled_edge_index, pooled_edge_attr, pooled_batch = adj_to_tg_batch(pooled_adj, self.detach_pool)

        return pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, loss, self.pool_assignment

    def losses(self):
        modularity_loss = sum([
            self.modularity_loss(self.pool_assignment[i].view(-1, self.pool_nodes), self.edge_index,
                                 self.edge_attr[:, i])
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
