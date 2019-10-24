import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import dense_diff_pool

# from torch_geometric.nn import GCNConv
from module import EGATConv
from utils import *
from torch import nn
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from scipy.sparse import coo_matrix


class Net(nn.Module):

    def __init__(self, writer, dropout=0.0):
        super(Net, self).__init__()

        self.in_channels = 7
        self.hidden_dim = 30
        self.pool_nodes = 16

        self.conv1 = nn.ModuleList([
            EGATConv(self.in_channels, self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim)
        ])

        self.pool_conv = nn.ModuleList([
            EGATConv(self.in_channels, self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            EGATConv(self.hidden_dim, self.pool_nodes)
        ])
        self.pool_fc = nn.Linear(2 * self.hidden_dim + self.pool_nodes, self.pool_nodes)

        self.conv2 = nn.ModuleList([
            EGATConv(self.hidden_dim, self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim)
        ])

        out_count = len(self.conv1) + len(self.conv2)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * out_count, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        adj = torch.stack([
            torch.tensor(to_scipy_sparse_matrix(data.edge_index, data.edge_attr, data.num_nodes).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)

        out_all = []

        # conv1
        alpha = edge_attr
        for conv_block in self.conv1:
            x, alpha = conv_block(x, edge_index, alpha)
            out_all.append(x)

        # soft pooling
        pool_conv_out_all = []
        alpha = edge_attr
        x = batch.x.to(self.device)  # orig x
        for conv_block in self.pool_conv:
            x, alpha = conv_block(x, edge_index, alpha)
            pool_conv_out_all.append(x)
        pool_conv_out_all = torch.cat(pool_conv_out_all, dim=1)
        assignment = self.pool_fc(pool_conv_out_all)

        x_conv1_out = out_all[-1]
        x_conv1_out = self.split_n(x_conv1_out, batch.num_graphs)
        assignment = self.split_n(assignment, batch.num_graphs)
        pooled_x, pooled_adj, link_loss, entropy_loss = dense_diff_pool(x_conv1_out, adj, assignment)

        # converting data
        data_list = []
        for i in range(batch.num_graphs):
            tmp_adj = pooled_adj[i]
            tmp_adj /= (adj.shape[-1] / pooled_adj.shape[-1]) ** 2  # normalize?
            edge_index, edge_attr = from_2d_tensor_adj(tmp_adj.clone())
            data_list.append(Data(edge_index=edge_index, edge_attr=edge_attr))
        tmp_batch = Batch.from_data_list(data_list)
        edge_index, edge_attr = tmp_batch.edge_index, tmp_batch.edge_attr
        pooled_x = pooled_x.reshape(-1, pooled_x.shape[-1])  # merge to batch
        pooled_x /= (adj.shape[-1] / pooled_adj.shape[-1])  # normalize?

        # conv2
        alpha = edge_attr
        x = pooled_x
        for conv_block in self.conv2:
            x, alpha = conv_block(x, edge_index, alpha)
            assert torch.isnan(x).sum() == 0
            out_all.append(x)

        # max pooling
        max_pooled_out_all = [torch.max(self.split_n(out, batch.num_graphs), dim=1)[0] for out in out_all]
        max_pooled_out_all = torch.cat(max_pooled_out_all, dim=-1)

        fc_out = self.fc(max_pooled_out_all)

        reg = link_loss + entropy_loss

        return fc_out, reg

    @property
    def device(self):
        return self.conv1[0].weight.device

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])


if __name__ == '__main__':
    from utils import gaussian_fit
    from dataset import ABIDE

    dataset = ABIDE(root='datasets/NYU', transform=gaussian_fit)
    model = Net()
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([data])
    model(batch)
    pass
