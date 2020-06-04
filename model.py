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
        self.pool1_nodes = 5
        self.first_attention_heads = 5
        self.conv_depth = 3  # 6 for single-site, 3 for multi-site (faster)
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
            nn.Sequential(nn.Linear(self.in_nodes, 10), nn.BatchNorm1d(10)),
            # nn.Sequential(nn.Linear(200, 10), nn.BatchNorm1d(10))
        ])

        self.cnp1 = ConvNPool(in_channels=self.in_channels,
                              pool_nodes=self.pool1_nodes,
                              attention_heads=self.first_attention_heads,
                              in_dims=1,
                              beta=1,
                              **cnp_params)

        self.domain_conv = ParallelResGraphConv(self.hidden_dim, self.hidden_dim, self.hidden_dim,
                                                dims=self.alpha_dim, depth=self.conv_depth)

        self.domain_fc = nn.Sequential(
            # nn.BatchNorm1d(self.conv_depth * self.hidden_dim * self.alpha_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(self.conv_depth * self.hidden_dim * self.alpha_dim * 2, 50),
            nn.ReLU(),
            # nn.BatchNorm1d(50),
            nn.Dropout(dropout),
            nn.Linear(50, 4),  # 20 sites
            nn.LogSoftmax(dim=-1)
        )

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1 * self.hidden_dim * self.alpha_dim * 1, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        all_x = [batch.x.to(self.device),
                 batch.adj_statistics.to(self.device),
                 batch.raw_adj.to(self.device), ]
        # batch.time_series.to(self.device)]
        edge_index, edge_attr = batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        num_graphs, batch_mask = batch.num_graphs, batch.batch.to(self.device)

        x = torch.cat([op(x) for op, x in zip(self.first_fc, all_x)], dim=-1)

        # domain_x_out = self.first_domain_fc(x.reshape(num_graphs, -1))

        # CNP
        cnp1_out_all, p1_x, p1_ei, p1_ea, p1_batch, p1_loss, p1_assignment = self.cnp1(x, edge_index, edge_attr, batch)
        reg = p1_loss.unsqueeze(0)
        # reg = torch.tensor(0.).to(self.device)

        # domain
        domain_out = self.domain_conv(torch.cat([x for _ in range(self.alpha_dim)], dim=-1),
                                      self.cnp1.alpha_index, self.cnp1.alpha, batch_mask)
        domain_out = torch.cat([torch.cat(d, dim=1) for d in domain_out], dim=-1)
        domain_out = domain_out.reshape(num_graphs, self.in_nodes, self.hidden_dim * self.conv_depth, self.alpha_dim)
        domain_out = torch.cat([domain_out.max(dim=1)[0], domain_out.mean(dim=1)], dim=-1)  # readout

        p1_x = p1_x.reshape(num_graphs, self.pool1_nodes, self.hidden_dim, self.alpha_dim)
        p1_x = p1_x.max(dim=1)[0]  # max pooling

        fc_out = self.final_fc(p1_x.reshape(num_graphs, -1))
        domain_fc_out = self.domain_fc(domain_out.reshape(num_graphs, -1))

        if self.logging_hist:
            self.writer.add_histogram('alpha1', self.cnp1.alpha.detach().cpu().flatten())
            self.writer.add_histogram('p1_ea', p1_ea.detach().cpu().flatten())
            self.writer.add_histogram('p1_assignment', p1_assignment.detach().cpu().flatten())
            adj_1 = batch_to_adj(self.cnp1.alpha_index, self.cnp1.alpha, 360, num_graphs)
            torch.save(adj_1, 'adj_1')
            fig_1 = plot_matrix(adj_1[0, 0].detach().cpu())
            fig_1.show()
            self.writer.add_figure('alpha1', fig_1)

        assert not nan_or_inf(fc_out)
        assert not nan_or_inf(domain_fc_out)

        return fc_out, domain_fc_out, reg

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
            nn.Linear(50, 20),
            nn.LogSoftmax(),
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        all_x = [batch.x.to(self.device),
                 batch.adj_statistics.to(self.device),
                 batch.raw_adj.to(self.device), ]
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

    dataset = ABIDE(root='datasets/ALL')
    model = Net()
    # model = MLP()
    data = dataset.__getitem__(0)
    batch = Batch.from_data_list([dataset.__getitem__(i) for i in range(2)])
    model(batch)
    pass
