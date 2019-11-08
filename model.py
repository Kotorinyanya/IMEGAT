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

EPS = 1e-15


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, hiddem_dim, out_channels, depth,
                 first_conv_layer=EGATConv,
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

    def forward(self, x, ei, ea, batch_mask):
        out_all = []

        # if self.block_chunk_size == 1:  # no batch norm
        #     for i, conv_block in enumerate(self.res_convs):
        #         if i >= 1:  # add res block
        #             x = torch.cat([out_all[-1], x], dim=-1)
        #         if conv_block.__class__ == EGATConv:  # EGATConv
        #             x, ea, ei = conv_block(x, ei, ea)
        #         else:  # GraphConv or etc.
        #             x = conv_block(x, ei, ea.view(-1))
        #         out_all.append(x)

        if self.block_chunk_size == 2:  # with batch norm
            for i, (conv_block, norm_block) in enumerate(chunks(self.res_convs, self.block_chunk_size)):
                if i >= 1:  # add res block
                    x = torch.cat([out_all[-1], x], dim=-1)
                if conv_block.__class__ == EGATConv:  # EGATConv
                    x, ea, ei = conv_block(x, ei, ea)
                else:  # GraphConv or etc.
                    x = conv_block(x, ei, ea.view(-1))
                # print(i, x[:5, :5])
                x = norm_block(x, batch_mask)
                out_all.append(x)

        return out_all

    @property
    def device(self):
        return self.res_convs[0].weight.device


class Pool(nn.Module):
    def __init__(self, in_channels, hiddem_dim, pool_nodes, depth):
        super(Pool, self).__init__()
        self.block_chunk_size = 2
        self.pool_nodes = pool_nodes
        self.detach_pool = False  # detach to train separately
        self.conv_depth = depth
        self.pool_convs = ResConvBlock(in_channels, hiddem_dim, pool_nodes, self.conv_depth,
                                       first_conv_layer=GraphConv,
                                       hidden_conv_layer=GraphConv,
                                       last_conv_layer=GraphConv)
        self.pool_fc = nn.Sequential(
            nn.Linear(hiddem_dim * (self.conv_depth - 1) + pool_nodes, 50),
            nn.Linear(50, pool_nodes)
        )

    def forward(self, x, edge_index, edge_attr, batch, x_to_pool):
        # convert edge_index to adj
        num_nodes = int(batch.num_nodes / batch.num_graphs)
        adj = torch.stack([
            torch.tensor(to_scipy_sparse_matrix(data.edge_index, data.edge_attr, num_nodes).todense(),
                         dtype=torch.float,
                         device=self.device)
            for data in batch.to_data_list()], dim=0)
        # variables
        batch_mask = batch.batch.to(self.device)
        num_graphs = batch.num_graphs

        # conv and fc
        pool_conv_out_all = self.pool_convs(x, edge_index, edge_attr, batch_mask)
        pool_conv_out_all = torch.cat(pool_conv_out_all, dim=1)
        assignment = self.pool_fc(pool_conv_out_all)
        # softmax
        pool_assignment = self.split_n(assignment, num_graphs)
        pool_assignment = torch.softmax(pool_assignment, dim=-1)
        # loss for S
        modularity_loss = self.modularity_loss(pool_assignment.view(-1, self.pool_nodes), edge_index, edge_attr)
        entropy_loss = self.entropy_loss(pool_assignment)
        link_loss = self.link_loss(pool_assignment, adj)
        # perform pooling
        pool_assignment = pool_assignment.detach() if self.detach_pool else pool_assignment
        pooled_x = pool_assignment.transpose(1, 2) @ self.split_n(x_to_pool, num_graphs)
        pooled_adj = pool_assignment.transpose(1, 2) @ adj @ pool_assignment

        # convert adj to edge_index
        data_list = []
        for i in range(num_graphs):
            tmp_adj = pooled_adj[i]
            tmp_adj /= (adj.shape[-1] / pooled_adj.shape[-1]) ** 2  # normalize?
            # tmp_adj.clone() to keep gradients
            edge_index, edge_attr = from_2d_tensor_adj(tmp_adj.detach() if self.detach_pool else tmp_adj.clone())
            tmp_data = Data(x=pooled_x[i], edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.pool_nodes)
            data_list.append(tmp_data)
        pooled_batch = Batch.from_data_list(data_list)
        # pooled_batch.to_data_list()
        pooled_edge_index, pooled_edge_attr = pooled_batch.edge_index, pooled_batch.edge_attr
        pooled_x = pooled_x.reshape(-1, pooled_x.shape[-1])  # merge to batch
        pooled_x /= (adj.shape[-1] / pooled_adj.shape[-1])  # normalize?

        return pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, entropy_loss, modularity_loss, link_loss

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])

    @staticmethod
    def modularity_loss(assignment, edge_index, edge_attr=None):
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


# noinspection PyTypeChecker


class Net(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(Net, self).__init__()

        self.in_channels = 11
        self.hidden_dim = 30
        self.in_nodes = 360
        # self.pool_percent = 0.25
        self.pool1_nodes = 16
        # self.pool2_nodes = 16
        self.beta_s = 2
        self.first_layer_heads = 1
        self.depth = 12
        self.first_layer_concat = False  # TODO: `True` is not implemented in ResConvBlock
        self.first_conv_out_size = self.hidden_dim * self.first_layer_heads \
            if self.first_layer_concat else self.hidden_dim

        partial_egat = partial(EGATConv, heads=self.first_layer_heads, concat=self.first_layer_concat)

        self.conv1 = ResConvBlock(self.in_channels, self.hidden_dim, self.hidden_dim, self.depth,
                                  first_conv_layer=partial_egat, hidden_conv_layer=GraphConv, last_conv_layer=GraphConv)
        self.pool1 = Pool(self.in_channels, self.hidden_dim, self.pool1_nodes, 4)
        self.conv2 = ResConvBlock(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.depth,
                                  first_conv_layer=partial_egat, hidden_conv_layer=GraphConv, last_conv_layer=GraphConv)
        # self.pool2 = Pool(self.hidden_dim * self.depth, self.hidden_dim, self.pool2_nodes, self.depth)
        # self.conv3 = ResConvBlock(self.hidden_dim * self.depth, self.hidden_dim, self.hidden_dim, self.depth,
        #                           first_conv_layer=egat)

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.first_conv_out_size * 2 + self.hidden_dim * (self.depth - 1) * 2, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 2)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr

        out_all = list()
        # conv-pool-conv
        out_all.append(self.conv1(x, edge_index, edge_attr, batch.batch.to(self.device)))
        p1_x, p1_ei, p1_ea, p1_batch, p1_el, p1_ml, p1_ll = self.pool1(x, edge_index, edge_attr, batch,
                                                                       out_all[-1][-1])
        # torch.cat(out_all[-1], dim=-1))
        out_all.append(self.conv2(p1_x, p1_ei, p1_ea, p1_batch.batch.to(self.device)))
        # p2_x, p2_ei, p2_ea, p2_batch, p2_el, p2_ml = self.pool2(p1_x, p1_ei, p1_ea, p1_batch,
        #                                                         torch.cat(out_all[-1], dim=-1))
        # out_all.append(self.conv3(p2_x, p2_ei, p2_ea, p2_batch.batch.to(self.device)))

        out_all = sum(out_all, [])  # !!!?

        # global pooling
        global_pooled_out_all = [torch.max(self.split_n(out, batch.num_graphs), dim=1)[0] for out in out_all]
        # global_pooled_out_all = [torch.mean(self.split_n(out, batch.num_graphs), dim=1) for out in out_all]
        global_pooled_out_all = torch.cat(global_pooled_out_all, dim=-1)

        reg = p1_el + p1_ml
        reg = reg.unsqueeze(0)

        fc_out = self.final_fc(global_pooled_out_all)

        return fc_out, reg

    @property
    def device(self):
        return self.conv1.device

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])


class ResGCN(nn.Module):
    def __init__(self, writer=None, dropout=0.0):
        super(ResGCN, self).__init__()

        self.in_channels = 11
        self.hidden_dim = 30
        self.in_nodes = 360
        self.block_chunk_size = 2
        self.first_layer_heads = 5
        self.first_layer_concat = False
        self.first_conv_out_size = self.hidden_dim * self.first_layer_heads \
            if self.first_layer_concat else self.hidden_dim

        self.convs = nn.ModuleList([
            EGATConv(self.in_channels, self.hidden_dim, heads=self.first_layer_heads, concat=self.first_layer_concat),
            InstanceNorm(self.first_conv_out_size),
            GraphConv(self.first_conv_out_size, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
            GraphConv(self.hidden_dim, self.hidden_dim),
            InstanceNorm(self.hidden_dim),
        ])

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.first_conv_out_size * 1 + self.hidden_dim * 9, 50),
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

        out_all = []

        # convs
        ea, ei = edge_attr, edge_index
        for conv_block, norm_block in chunks(self.convs, self.block_chunk_size):
            if conv_block.__class__ == EGATConv:
                x, ea, ei = conv_block(x, ei, ea)
            else:  # GraphConv etc.
                x = conv_block(x, ei, ea.view(-1))
            x = norm_block(x, batch_mask)
            # alpha_reg_all.append(entropy(a, ei).mean())
            out_all.append(x)

        # global pooling
        global_pooled_out_all = [torch.max(self.split_n(out, batch.num_graphs), dim=1)[0] for out in out_all]
        # global_pooled_out_all = [torch.mean(self.split_n(out, batch.num_graphs), dim=1) for out in out_all]
        global_pooled_out_all = torch.cat(global_pooled_out_all, dim=-1)

        fc_out = self.fc(global_pooled_out_all)

        reg = torch.tensor([0.], device=self.device)

        return fc_out, reg

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])

    @property
    def device(self):
        return self.convs[0].weight.device


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
