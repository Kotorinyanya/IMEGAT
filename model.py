# from torch_geometric.nn import GCNConv
from functools import partial
from module import *


# noinspection PyTypeChecker


class Net(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(Net, self).__init__()

        self.in_channels = 7
        self.hidden_dim = 30
        self.in_nodes = 360
        # self.pool_percent = 0.25
        self.pool1_nodes = 10
        # self.pool2_nodes = 10
        # self.pool3_nodes = 6
        self.first_attention_heads = 5
        self.conv_depth = 3
        self.pool_conv_depth = 3
        self.att_dropout = 0.
        self.concat = True
        self.alpha_dim = self.first_attention_heads if self.concat else 1
        self.first_conv_out_size = self.hidden_dim * self.alpha_dim

        cnp_params = {"hidden_dim": self.hidden_dim,
                      "out_channels": self.hidden_dim,
                      "concat": self.concat,
                      "att_dropout": self.att_dropout,
                      "conv_depth": self.conv_depth,
                      "pool_conv_depth": self.pool_conv_depth,
                      "ml": 1,
                      "ll": 1}

        self.cnp1 = ConvNPool(in_channels=self.in_channels,
                              pool_nodes=self.pool1_nodes,
                              attention_heads=self.first_attention_heads,
                              in_dims=1,
                              el=1,
                              **cnp_params)
        self.conv2 = ConvNPool(in_channels=self.hidden_dim,
                               attention_heads=1,
                               in_dims=self.first_attention_heads,
                               no_pool=True,
                               **cnp_params)
        # self.cnp3 = ConvNPool(in_channels=self.hidden_dim,
        #                       pool_nodes=self.pool3_nodes,
        #                       attention_heads=1,
        #                       in_dims=self.first_attention_heads,
        #                       **cnp_params)

        # self.ins_norm = nn.ModuleList([
        #     InstanceNorm(self.hidden_dim),
        #     InstanceNorm(self.hidden_dim)
        # ])

        self.final_fc = nn.Sequential(
            # nn.BatchNorm1d(self.pool3_nodes * self.hidden_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(self.pool1_nodes * self.hidden_dim * self.alpha_dim * 1, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(100, 2),
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr
        num_graphs = batch.num_graphs

        # CNP
        cnp1_out_all, p1_x, p1_ei, p1_ea, p1_batch, p1_loss, p1_assignment = self.cnp1(x, edge_index, edge_attr, batch)
        conv_out_2 = self.conv2(p1_x, p1_ei, p1_ea, p1_batch)
        # cnp3_out_all, p3_x, p3_ei, p3_ea, p3_batch, p3_loss, p3_assignment = self.cnp3(p2_x, p2_ei, p2_ea, p2_batch)
        reg = p1_loss
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

        fc_out = self.final_fc(conv_out_2.reshape(num_graphs, -1))

        return fc_out, reg

    @property
    def device(self):
        return self.cnp1.device

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])


class CPNet(nn.Module):

    def __init__(self, writer=None, dropout=0.0):
        super(CPNet, self).__init__()

        self.in_channels = 11
        self.hidden_dim = 10
        self.in_nodes = 360
        self.pool1_nodes = 90
        self.pool2_nodes = 22
        # self.pool3_nodes = 8
        self.first_layer_heads = 5
        self.depth = 1
        self.pool_depth = 2
        self.first_layer_concat = False  # TODO: `True` is not implemented in ResConvBlock
        self.first_conv_out_size = self.hidden_dim * self.first_layer_heads \
            if self.first_layer_concat else self.hidden_dim

        self.egat = EGATConv(self.in_channels, self.hidden_dim, heads=self.first_layer_heads,
                             concat=self.first_layer_concat, att_dropout=0)
        self.conv1 = ResConvBlock(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.depth)
        self.pool1 = Pool(self.in_channels, self.hidden_dim, self.pool1_nodes, self.pool_depth, dims=5)
        self.conv2 = ResConvBlock(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.depth)
        self.pool2 = Pool(self.hidden_dim, self.hidden_dim, self.pool2_nodes, self.pool_depth)
        # self.conv3 = ResConvBlock(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.depth,
        #                           first_conv_layer=partial_egat)
        # self.pool3 = Pool(self.hidden_dim, self.hidden_dim, self.pool3_nodes, self.pool_depth)

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.pool2_nodes * self.hidden_dim, 50),
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
        out_all.append(self.conv2(p1_x, p1_ei, p1_ea, p1_batch.batch.to(self.device)))
        p2_x, p2_ei, p2_ea, p2_batch, p2_el, p2_ml, p2_ll = self.pool2(p1_x, p1_ei, p1_ea, p1_batch,
                                                                       out_all[-1][-1])
        # out_all.append(self.conv3(p2_x, p2_ei, p2_ea, p2_batch.batch.to(self.device)))
        # p3_x, p3_ei, p3_ea, p3_batch, p3_el, p3_ml, p3_ll = self.pool3(p2_x, p2_ei, p2_ea, p2_batch,
        #                                                                out_all[-1][-1])

        out_all = sum(out_all, [])  # !!!?

        pooled_out_all = p2_x.reshape(batch.num_graphs, -1)
        # print(p3_x[:, 0])

        reg = p1_ml + p1_el + p2_ml + p2_el
        # reg = torch.tensor([0.], device=self.device)
        reg = reg.unsqueeze(0)

        fc_out = self.final_fc(pooled_out_all)

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
        self.first_layer_heads = 5
        self.first_layer_concat = False
        self.depth = 6
        self.first_conv_out_size = self.hidden_dim * self.first_layer_heads \
            if self.first_layer_concat else self.hidden_dim

        partial_egat = partial(EGATConv, heads=self.first_layer_heads, concat=self.first_layer_concat,
                               att_dropout=dropout)

        self.convs = ResConvBlock(self.in_channels, self.hidden_dim, self.hidden_dim, self.depth,
                                  first_conv_layer=GraphConv, hidden_conv_layer=GraphConv, last_conv_layer=GraphConv)

        self.final_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear((self.first_conv_out_size + self.hidden_dim * (self.depth - 1)) * 1, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 2)
        )

    def forward(self, batch):
        if type(batch) == list:  # Data list
            batch = Batch.from_data_list(batch)

        x, edge_index, edge_attr = batch.x.to(self.device), batch.edge_index.to(self.device), batch.edge_attr
        edge_attr = edge_attr.to(self.device) if edge_attr is not None else edge_attr

        out_all = self.convs(x, edge_index, edge_attr, batch.batch.to(self.device))

        # global pooling
        global_pooled_out_all = [torch.max(self.split_n(out, batch.num_graphs), dim=1)[0] for out in out_all]
        # global_pooled_out_all = [torch.mean(self.split_n(out, batch.num_graphs), dim=1) for out in out_all]
        global_pooled_out_all = torch.cat(global_pooled_out_all, dim=-1)

        # reg = p1_el + p1_ml
        reg = torch.tensor([0.], device=self.device)
        reg = reg.unsqueeze(0)

        fc_out = self.final_fc(global_pooled_out_all)

        return fc_out, reg

    @staticmethod
    def split_n(tensor, n):
        return tensor.reshape(n, int(tensor.shape[0] / n), tensor.shape[1])

    @property
    def device(self):
        return self.convs.device


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
    batch = Batch.from_data_list([dataset.__getitem__(i) for i in range(10)])
    model(batch)
    pass
