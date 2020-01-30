# from torch_geometric.nn import GCNConv
from functools import partial
from module import *
from nilearn.plotting import plot_matrix


# noinspection PyTypeChecker


class Net(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(Net, self).__init__()

        self.writer = writer
        self.in_channels = 30
        self.hidden_dim = 30
        self.in_nodes = 360
        # self.pool_percent = 0.25
        self.pool1_nodes = 22
        self.pool2_nodes = 5
        # self.pool3_nodes = 6
        self.first_attention_heads = 5
        self.conv_depth = 6
        self.pool_conv_depth = 3
        self.att_dropout = 0.
        self.concat = True
        self.alpha_dim = self.first_attention_heads if self.concat else 1
        self.first_conv_out_size = self.hidden_dim * self.alpha_dim

        self.logging_hist = False

        cnp_params = {"hidden_dim": self.hidden_dim,
                      "out_channels": self.hidden_dim,
                      "concat": self.concat,
                      "att_dropout": self.att_dropout,
                      "conv_depth": self.conv_depth,
                      "pool_conv_depth": self.pool_conv_depth,
                      "ml": 1,
                      "ll": 0,
                      "el": 1}

        self.first_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(7, 10), nn.BatchNorm1d(10)),
            nn.Sequential(nn.Linear(4, 10), nn.BatchNorm1d(10)),
            nn.Sequential(nn.Linear(360, 10), nn.BatchNorm1d(10)),
            # nn.Sequential(nn.Linear(200, 10), nn.BatchNorm1d(10))
        ])

        self.cnp1 = ConvNPool(in_channels=self.in_channels,
                              pool_nodes=self.pool2_nodes,
                              attention_heads=self.first_attention_heads,
                              in_dims=1,
                              beta=1,
                              **cnp_params)
        # self.conv2 = ConvNPool(in_channels=self.hidden_dim,
        #                        attention_heads=1,
        #                        in_dims=self.first_attention_heads,
        #                        no_pool=True,
        #                        **cnp_params)
        # self.cnp2 = ConvNPool(in_channels=self.hidden_dim,
        #                       pool_nodes=self.pool2_nodes,
        #                       attention_heads=1,
        #                       in_dims=self.first_attention_heads,
        #                       **cnp_params)

        # self.ins_norm = nn.ModuleList([
        #     InstanceNorm(self.hidden_dim),
        #     InstanceNorm(self.hidden_dim)
        # ])

        # self.alpha_conv_weight = Parameter(torch.ones(self.alpha_dim, 1) / self.alpha_dim)
        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1 * self.hidden_dim * self.alpha_dim * 1, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        all_x = [batch.x.to(self.device),
                 batch.adj_statistics.to(self.device),
                 batch.raw_adj.to(self.device),]
                 # batch.time_series.to(self.device)]
        edge_index, edge_attr = batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        num_graphs, batch_mask = batch.num_graphs, batch.batch.to(self.device)

        x = torch.cat([op(x) for op, x in zip(self.first_fc, all_x)], dim=-1)

        # CNP
        cnp1_out_all, p1_x, p1_ei, p1_ea, p1_batch, p1_loss, p1_assignment = self.cnp1(x, edge_index, edge_attr, batch)
        # p1_x = self.cnp1(x, edge_index, edge_attr, batch)
        # conv_out_2 = self.conv2(p1_x, p1_ei, p1_ea, p1_batch)
        # cnp2_out_all, p2_x, p2_ei, p2_ea, p2_batch, p2_loss, p2_assignment = self.cnp2(p1_x, p1_ei, p1_ea, p1_batch)
        reg = p1_loss
        # reg = torch.tensor(0.).to(self.device)
        reg = reg.unsqueeze(0)

        # x1 = p3_assignment.transpose(-2, -1).detach() @ \
        #      p2_assignment.transpose(-2, -1).detach() @ \
        #      p1_x.reshape(num_graphs, -1, self.hidden_dim, self.alpha_dim).permute(3, 0, 1, 2) / \
        #      (self.pool1_nodes / self.pool3_nodes)  # normalize
        # x1 = x1.permute(1, 2, 3, 0)
        # x2 = p3_assignment.transpose(-2, -1).detach() @ \
        #      p2_x.reshape(num_graphs, -1, self.hidden_dim, self.alpha_dim).permute(3, 0, 1, 2) / \
        #      (self.pool2_nodes / self.pool3_nodes)  # normalize
        # x2 = x2.permute(1, 2, 3, 0)
        # x3 = p3_x
        # all_pooled_x = [x1, x2, x3]
        # for i, x in enumerate(all_pooled_x):
        #     x = x.reshape(x3.shape[0], -1)
        #     x = self.ins_norm[i](x, p3_batch.batch.to(self.device))
        #     all_pooled_x[i] = x.reshape(num_graphs, -1)
        # all_pooled_x = torch.cat(all_pooled_x, dim=-1)

        p1_x = p1_x.reshape(num_graphs, self.pool2_nodes, self.hidden_dim, self.alpha_dim)
        p1_x = p1_x.max(dim=1)[0]  # max pooling
        # # p1_x = p1_x @ self.alpha_conv_weight
        fc_out = self.final_fc(p1_x.reshape(num_graphs, -1))

        # if self.logging_hist:
        # self.writer.add_histogram('alpha1', self.cnp1.alpha.detach().cpu().flatten())
        # self.writer.add_histogram('alpha2', self.cnp2.alpha.detach().cpu().flatten())
        # self.writer.add_histogram('p1_ea', p1_ea.detach().cpu().flatten())
        # self.writer.add_histogram('p2_ea', p2_ea.detach().cpu().flatten())
        # self.writer.add_histogram('p1_assignment', p1_assignment.detach().cpu().flatten())
        # self.writer.add_histogram('p2_assignment', p2_assignment.detach().cpu().flatten())
        # adj_1 = batch_to_adj(self.cnp1.alpha_index, self.cnp1.alpha, 360, num_graphs)
        # torch.save(adj_1, 'adj_1')
        # adj_2 = batch_to_adj(self.cnp2.alpha_index, self.cnp2.alpha, self.pool1_nodes, num_graphs)
        # fig_1 = plot_matrix(adj_1[0, 0].detach().cpu())
        # fig_2 = plot_matrix(adj_2[0, 0].detach().cpu())
        # fig_1.show()
        # fig_2.show()
        # self.writer.add_figure('alpha1', fig_1)
        # self.writer.add_figure('alpha2', fig_2)

        assert not nan_or_inf(fc_out)

        return fc_out, reg

    @property
    def device(self):
        return self.cnp1.device

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])


class ResGCN(nn.Module):
    def __init__(self, writer=None, dropout=0.0):
        super(ResGCN, self).__init__()

        self.writer = writer
        self.in_channels = 7
        self.hidden_dim = 10
        self.in_nodes = 360
        # self.pool_percent = 0.25
        self.pool1_nodes = 22
        self.pool2_nodes = 4
        # self.pool3_nodes = 6
        self.first_attention_heads = 5
        self.conv_depth = 3
        self.pool_conv_depth = 3
        self.att_dropout = 0.
        self.concat = True
        self.alpha_dim = self.first_attention_heads if self.concat else 1
        self.first_conv_out_size = self.hidden_dim * self.alpha_dim

        self.logging_hist = False

        cnp_params = {"hidden_dim": self.hidden_dim,
                      "out_channels": self.hidden_dim,
                      "concat": self.concat,
                      "att_dropout": self.att_dropout,
                      "conv_depth": self.conv_depth,
                      "pool_conv_depth": self.pool_conv_depth,
                      "ml": 0,
                      "ll": 0,
                      "el": 0}

        self.cnp1 = ConvNPool(in_channels=self.in_channels,
                              pool_nodes=self.pool2_nodes,
                              attention_heads=self.first_attention_heads,
                              in_dims=1,
                              beta=2,
                              no_pool=True,
                              **cnp_params)

        # self.alpha_conv_weight = Parameter(torch.ones(self.alpha_dim, 1) / self.alpha_dim)
        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * self.hidden_dim * self.alpha_dim * 1, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        num_graphs = batch.num_graphs

        # CNP
        out = self.cnp1(x, edge_index, edge_attr, batch)
        reg = torch.tensor(0.).to(self.device)
        reg = reg.unsqueeze(0)

        out = out.reshape(num_graphs, self.in_nodes, self.hidden_dim, self.alpha_dim)
        out_max = out.max(dim=1)[0]  # pooling
        out_min = out.min(dim=1)[0]
        out = torch.cat([out_max, out_min], dim=-1)

        fc_out = self.final_fc(out.reshape(num_graphs, -1))

        return fc_out, reg

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])

    @property
    def device(self):
        return self.cnp1.device


class MLP(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(MLP, self).__init__()

        self.first_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(7, 10), nn.BatchNorm1d(10)),
            nn.Sequential(nn.Linear(4, 10), nn.BatchNorm1d(10)),
            nn.Sequential(nn.Linear(360, 10), nn.BatchNorm1d(10)),
            # nn.Sequential(nn.Linear(200, 10), nn.BatchNorm1d(10))
        ])

        self.fc = nn.Sequential(
            nn.Linear(30 * 360, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 2),
            nn.LogSoftmax(),
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        all_x = [batch.x.to(self.device),
                 batch.adj_statistics.to(self.device),
                 batch.raw_adj.to(self.device),]
                 # batch.time_series.to(self.device)]
        edge_index, edge_attr = batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        num_graphs = batch.num_graphs
        num_nodes = int(batch.num_nodes / batch.num_graphs)

        in_x = torch.cat([op(x) for op, x in zip(self.first_fc, all_x)], dim=-1)
        in_x = in_x.reshape(num_graphs, -1)

        out = self.fc(in_x)

        reg = torch.tensor(0.).to(self.device)

        return out, reg

    @property
    def device(self):
        return self.fc[0].weight.device


if __name__ == '__main__':
    from utils import gaussian_fit
    from dataset import ABIDE

    dataset = ABIDE(root='datasets/NYU')
    # model = Net()
    model = MLP()
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([dataset.__getitem__(i) for i in range(2)])
    model(batch)
    pass
