from torch_geometric.data import Batch, Data
from torch_geometric.nn import dense_diff_pool

# from torch_geometric.nn import GCNConv
from module import EGATConv
from utils import *
from torch import nn
from torch_geometric.utils import to_scipy_sparse_matrix
from torch.nn import BatchNorm1d
from module import InstanceNorm

EPS = 1e-15


class Net(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(Net, self).__init__()

        self.in_channels = 11
        self.hidden_dim = 30
        self.in_nodes = 360
        self.pool_nodes = 16
        self.beta_s = 2
        self.block_chunk_size = 2
        self.first_layer_heads = 5
        self.first_layer_concat = False
        self.first_conv_out_size = self.hidden_dim * self.first_layer_heads \
            if self.first_layer_concat else self.hidden_dim

        self.conv1 = nn.ModuleList([
            EGATConv(self.in_channels, self.hidden_dim, heads=self.first_layer_heads, concat=self.first_layer_concat),
            InstanceNorm(self.first_conv_out_size),
            EGATConv(self.first_conv_out_size, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
        ])

        self.pool_conv = nn.ModuleList([
            EGATConv(self.in_channels, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            EGATConv(self.hidden_dim, self.pool_nodes),
            InstanceNorm(self.pool_nodes),
        ])
        self.pool_fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim + self.pool_nodes, 50),
            InstanceNorm(50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, self.pool_nodes)
        )
        self.pool_s = nn.Parameter(torch.Tensor(self.in_nodes, self.pool_nodes))
        self.pool_s.data = nn.init.xavier_uniform_(self.pool_s.data, gain=nn.init.calculate_gain('relu'))

        self.conv2 = nn.ModuleList([
            EGATConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            EGATConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
        ])

        self.fc = nn.Sequential(
            nn.Linear(self.first_conv_out_size * 1 + self.hidden_dim * 5, 50),
            InstanceNorm(50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 2)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        batch_mask = batch.batch.to(self.device)
        adj = torch.stack([
            torch.tensor(to_scipy_sparse_matrix(data.edge_index, data.edge_attr, data.num_nodes).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)

        out_all = []
        # alpha_reg_all = []

        # conv1
        a, ei = edge_attr, edge_index
        for conv_block, norm_block in chunks(self.conv1, self.block_chunk_size):
            x, a, ei, ea = conv_block(x, ei, a)
            x = norm_block(x, batch_mask)
            # alpha_reg_all.append(entropy(a, ei).mean())
            out_all.append(x)

        # soft pooling
        pool_conv_out_all = []
        a, ei = edge_attr, edge_index
        x = batch.x.to(self.device)  # orig x
        for conv_block, norm_block in chunks(self.pool_conv, self.block_chunk_size):
            x, a, ei, ea = conv_block(x, ei, a)
            x = norm_block(x, batch_mask)
            # alpha_reg_all.append(entropy(a, ei).mean())
            pool_conv_out_all.append(x)
        pool_conv_out_all = torch.cat(pool_conv_out_all, dim=1)
        assignment = self.pool_fc(pool_conv_out_all)
        assignment = self.split_n(assignment, batch.num_graphs)
        pool_assignment = (1 / self.beta_s) * assignment + \
                          (self.beta_s - 1 / self.beta_s) * self.pool_s
        pool_assignment = torch.softmax(pool_assignment, dim=-1)

        # modularity loss
        modularity_loss = self.modularity_reg(pool_assignment.view(-1, self.pool_nodes), edge_index, edge_attr)
        # entropy loss
        entropy_loss = (-pool_assignment * torch.log(pool_assignment + EPS)).sum(dim=-1).mean()
        pooled_x, pooled_adj = self.dense_diff_pool(self.split_n(out_all[-1], batch.num_graphs),
                                                                        adj, pool_assignment)
        reg = entropy_loss + modularity_loss

        # converting data
        data_list = []
        for i in range(batch.num_graphs):
            tmp_adj = pooled_adj[i]
            # tmp_adj /= (adj.shape[-1] / pooled_adj.shape[-1]) ** 2  # normalize?
            # tmp_adj[tmp_adj == 0] = 1e-16  # fully connected
            edge_index, edge_attr = from_2d_tensor_adj(tmp_adj.clone())
            data_list.append(Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.pool_nodes))
        p1_batch = Batch.from_data_list(data_list)
        edge_index, edge_attr = p1_batch.edge_index, p1_batch.edge_attr
        p1_batch_mask = p1_batch.batch.to(self.device)
        pooled_x = pooled_x.reshape(-1, pooled_x.shape[-1])  # merge to batch
        # pooled_x /= (adj.shape[-1] / pooled_adj.shape[-1])  # normalize?

        # conv2
        a, ei = edge_attr, edge_index
        x = pooled_x
        for conv_block, norm_block in chunks(self.conv2, self.block_chunk_size):
            x, a, ei, ea = conv_block(x, ei, a)
            x = norm_block(x, p1_batch_mask)
            # alpha_reg_all.append(entropy(a, ei).mean())
            out_all.append(x)

        # global pooling
        global_pooled_out_all = [torch.max(self.split_n(out, batch.num_graphs), dim=1)[0] for out in out_all]
        # global_pooled_out_all = [torch.mean(self.split_n(out, batch.num_graphs), dim=1) for out in out_all]
        global_pooled_out_all = torch.cat(global_pooled_out_all, dim=-1)

        fc_out = self.fc(global_pooled_out_all)

        # reg += torch.mean(torch.stack(alpha_reg_all))
        reg = reg.unsqueeze(0)

        return fc_out, reg

    @property
    def device(self):
        return self.conv1[0].weight.device

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])

    @staticmethod
    def modularity_reg(assignment, edge_index, edge_attr=None):
        """
        continues version of negative modularity (with positive value)
        :return:
        """
        edge_attr = 1 if edge_attr is None else edge_attr
        row, col = edge_index
        reg = (edge_attr * torch.pow((assignment[row] - assignment[col]), 2).sum(1)).mean()
        return reg

    @staticmethod
    def dense_diff_pool(x, adj, s, mask=None):
        r"""Differentiable pooling operator from the `"Hierarchical Graph
        Representation Learning with Differentiable Pooling"
        <https://arxiv.org/abs/1806.08804>`_ paper

        .. math::
            \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
            \mathbf{X}

            \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
            \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

        based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
        \times N \times C}`.
        Returns pooled node feature matrix, coarsened adjacency matrix and the
        auxiliary link prediction objective :math:`\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) \cdot {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|_F`.

        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}` with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature dimension
                :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`.
            s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
                \times N \times C}` with number of clusters :math:`C`. The softmax
                does not have to be applied beforehand, since it is executed
                within this method.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        return out, out_adj


class MLP(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(360 * 11, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)
        x = batch.x.to(self.device).float()
        # edge_attr = batch.edge_attr.to(self.device).float()
        x = x.reshape(batch.num_graphs, -1)
        # x = torch.cat([x, edge_attr.reshape(batch.num_graphs, -1)], dim=-1)
        # x = edge_attr.reshape(batch.num_graphs, -1)
        out = self.fc(x)

        reg = torch.tensor([0.]).to(self.device)

        return out, reg

    @property
    def device(self):
        return self.fc[0].weight.device


if __name__ == '__main__':
    from utils import gaussian_fit
    from dataset import ABIDE

    dataset = ABIDE(root='datasets/NYU', transform=z_score_norm_data)
    model = Net()
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([data])
    model(batch)
    pass
