import logging
import os.path as osp

import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
import socket
from datetime import datetime
import networkx as nx


def use_logging(level='info'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == 'warn':
                logging.warning("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args)

        return wrapper

    return decorator


def get_model_log_dir(comment, model_name):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = osp.join(
        current_time + '_' + socket.gethostname() + '_' + comment + '_' + model_name)
    return log_dir


def to_cuda(data_list, device):
    for i, data in enumerate(data_list):
        for k, v in data:
            data[k] = v.to(device)
        data_list[i] = data
    return data_list


def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes=None):
    dtype, device = edge_index.dtype, edge_index.device
    loop = torch.arange(0, num_nodes, dtype=dtype, device=device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)
    ones = torch.ones([edge_index.shape[1] - edge_attr.shape[0], edge_attr.shape[1]], dtype=edge_attr.dtype,
                      device=edge_attr.device)
    edge_attr = torch.cat([edge_attr, ones], dim=0)
    assert edge_index.shape[1] == edge_attr.shape[0]
    return edge_index, edge_attr


def z_score_norm(tensor, mean=None, std=None):
    """
    Normalize a tensor with mean and standard deviation.
    Args:
        tensor (Tensor): Tensor image of size [num_nodes, num_node_features] to be normalized.

    Returns:
        Tensor: Normalized tensor.
    """
    mean = tensor.mean(dim=0) if mean is None else mean
    std = tensor.std(dim=0) if std is None else std
    std[std == 0] = 1e-6
    normed_tensor = (tensor - mean) / std
    return normed_tensor


def z_score_norm_data(data):
    data.x = z_score_norm(data.x)
    return data


def log_along_dim(tensor, log_dim):
    tensor[:, log_dim] = torch.log(tensor[:, log_dim])
    tensor[tensor != tensor] = 0  # nan
    tensor[tensor.eq(float('-inf'))] = 0  # -inf
    return tensor


def custom_norm(tensor):
    """
    0-6: ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv']
    :param tensor:
    :return:
    """
    # missing value in dim 2
    tensor[:, 2][tensor[:, 2] == 0] = .9
    # log dim
    log_dim = [0, 1, 2, 4, 6]
    tensor = log_along_dim(tensor, log_dim)
    tensor = z_score_norm(tensor)
    return tensor


def custom_norm_data(data):
    data.x = custom_norm(data.x)
    return data


def new_ones(data):
    data.x = torch.ones_like(data.x)
    return data


def gaussian_fit(data):
    data.x = data.x.normal_(mean=0, std=1)
    return data


def z_score_over_node(x, mean, std):
    """

    :param x: Tensor (num_graphs, num_nodes, num_features)
    :param mean: Tensor (num_nodes, num_features)
    :param std:
    :return:
    """
    EPS = 1e-5
    std += EPS

    num_nodes = x.shape[1]
    for i in range(num_nodes):
        x[:, i, :] = (x[:, i, :] - mean[i]) / std[i]
    return x


def norm_train_val(dataset, train_idx, test_idx, num_nodes=360):
    tensor = dataset.data.x
    # missing value in dim 2
    # tensor[:, 2][tensor[:, 2] == 0] = .99
    # log along dim
    tensor = log_along_dim(tensor, [6])
    dataset.data.x = tensor

    train_dataset = dataset.copy(train_idx)
    test_dataset = dataset.copy(test_idx)
    feature_dim = train_dataset.data.x.shape[-1]
    train_x = train_dataset.data.x.reshape(-1, num_nodes, feature_dim)
    test_x = test_dataset.data.x.reshape(-1, num_nodes, feature_dim)

    train_mean = train_x.mean(0)
    train_std = test_x.std(0)

    train_x = z_score_over_node(train_x, train_mean, train_std)
    test_x = z_score_over_node(test_x, train_mean, train_std)

    train_dataset.data.x = train_x.reshape(-1, feature_dim)
    test_dataset.data.x = test_x.reshape(-1, feature_dim)

    return train_dataset, test_dataset


def doubly_stochastic_normalization_2d_tensor(adj):
    """

    :param adj: 2d Tensor
    :return:
    """
    num_nodes = adj.shape[0]
    tilde_adj = adj / adj.sum(1).reshape(-1, 1)
    col_sums = tilde_adj.sum(0)
    normed_adj = torch.zeros_like(adj)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            normed_adj[i, j] = (tilde_adj[i] * tilde_adj[j] / col_sums).sum()
    # to symmetric
    normed_adj = torch.max(normed_adj, normed_adj.t())
    return normed_adj


def positive_transform(conn_matrix):
    return 1 - np.sqrt((1 - conn_matrix) / 2)


def transform_e(adj):
    adj = positive_transform(adj)
    return adj


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


from torch_scatter import scatter_max, scatter_add


def real_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    src = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = src.exp()
    assert not nan_or_inf(out)
    oout = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    assert not nan_or_inf(oout)

    return oout


def norm_edge_attr(edge_index, num_nodes, edge_weight, type=1, improved=False, dtype=None):
    # fill_value = 1 if not improved else 2
    # edge_index, edge_weight = add_remaining_self_loops(
    #     edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    if type == 1:
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif type == 2:
        deg_inv = deg.pow(-1)
        norm = deg_inv[row] * edge_weight

    return edge_index, norm


def entropy(src, edge_index, num_nodes=None):
    EPS = 1e-32
    index, _ = edge_index
    num_nodes = maybe_num_nodes(index, num_nodes)

    # out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = -src * torch.log(src + EPS)
    out = scatter_add(out, index, dim=0, dim_size=num_nodes)

    return out


def add_self_loops_mul(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        loop_weight = edge_weight.new_full((num_nodes, edge_weight.shape[1]), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def from_2d_tensor_adj(adj):
    """
    maintain gradients
    Args:
        A : Tensor (num_nodes, num_nodes)
    """
    assert adj.dim() == 2
    edge_index = adj.nonzero().t().detach()
    row, col = edge_index
    edge_weight = adj[row, col]

    return edge_index, edge_weight


def from_3d_tensor_adj(adj):
    """
    maintain gradients
    Args:
        A : Tensor (dims, num_nodes, num_nodes)
    """
    assert adj.dim() == 3
    # return list(zip(*[from_2d_tensor_adj(adj[k]) for k in range(adj.shape[0])]))
    edge_index = adj.nonzero()[:, 1:].unique(dim=0).t().detach()  # make sure edge_index is same along axises
    row, col = edge_index
    edge_weight = adj.permute(1, 2, 0)[row, col]
    return edge_index, edge_weight


def my_to_data_list(batch, num_nodes):
    r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
    from the batch object.
    The batch object must have been created via :meth:`from_data_list` in
    order to be able reconstruct the initial objects."""

    if batch.__slices__ is None:
        raise RuntimeError(
            ('Cannot reconstruct data list from batch because the batch '
             'object was not created using Batch.from_data_list()'))

    keys = [key for key in batch.keys if key[-5:] != 'batch']
    cumsum = {key: 0 for key in keys}
    data_list = []
    for i in range(len(batch.__slices__[keys[0]]) - 1):
        data = batch.__data_class__()
        for key in keys:
            data[key] = batch[key].narrow(
                data.__cat_dim__(key, batch[key]), batch.__slices__[key][i],
                batch.__slices__[key][i + 1] - batch.__slices__[key][i])
            data[key] = data[key] - cumsum[key]
            cumsum[key] += num_nodes
        data_list.append(data)

    return data_list


def batch_to_adj(edge_index, edge_attr, num_nodes, num_graphs):
    """

    :param edge_index:
    :param edge_attr: Tensor with shape (num_edges, dim)
    :param num_nodes:
    :param num_graphs:
    :return: adj with shape (dim, num_graphs, num_nodes, num_nodes)
    """
    assert edge_attr.dim() == 2
    dims = edge_attr.shape[1]

    adjs = []
    for i in range(num_graphs):
        start, end = i * num_nodes, (i + 1) * num_nodes
        mask = ((edge_index >= start) & (edge_index < end))[0]
        edge_index_i = edge_index.t()[mask] - start  # starts from 0
        row, col = edge_index_i.t()
        edge_attr_i = edge_attr[mask]
        adj_i = torch.zeros((num_nodes, num_nodes, dims)).to(edge_attr.device)
        adj_i[row, col] = edge_attr_i
        adj_i = adj_i.permute(2, 0, 1)
        adjs.append(adj_i)
    adj = torch.stack(adjs, dim=1)

    return adj


def adj_to_batch(adj):
    """

    :param adj: Tensor with shape (dim, num_graphs, num_nodes, num_nodes)
    :return: edge_index, edge_attr, batch_mask
    """
    assert adj.dim() == 4
    num_graphs = adj.shape[1]
    num_nodes = adj.shape[2]

    edge_index = torch.tensor([], device=adj.device, dtype=torch.long)
    edge_attr = torch.tensor([], device=adj.device, dtype=adj.dtype)
    for i in range(num_graphs):
        adj_i = adj[:, i, :, :]
        edge_index_i, edge_attr_i = from_3d_tensor_adj(adj_i)
        edge_index_i += i * num_nodes  # batch code
        edge_index = torch.cat([edge_index, edge_index_i], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr_i], dim=0)

    batch_mask = torch.tensor(sum([[i] * num_nodes for i in range(num_graphs)], []), device=adj.device)

    return edge_index, edge_attr, batch_mask


def adj_to_tg_batch(adj, detach_pool=False):
    """

    :param detach_pool:
    :param adj: Tensor with shape (dim, num_graphs, num_nodes, num_nodes)
    :return:
    """
    assert adj.dim() == 4
    num_graphs = adj.shape[1]
    num_nodes = adj.shape[2]

    data_list = []
    for i in range(num_graphs):
        tmp_adj = adj[:, i, :, :]
        edge_index, edge_attr = from_3d_tensor_adj(tmp_adj.detach() if detach_pool else tmp_adj.clone())
        tmp_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        data_list.append(tmp_data)
    pooled_batch = Batch.from_data_list(data_list)
    pooled_edge_index, pooled_edge_attr = pooled_batch.edge_index, pooled_batch.edge_attr

    return pooled_edge_index, pooled_edge_attr, pooled_batch


def nan_or_inf(x):
    return torch.isnan(x).any() or x.eq(float('inf')).any() or x.eq(float('-inf')).any()


def check_strongly_connected(adj):
    G = nx.from_numpy_array(adj)
    G = G.to_directed()
    return nx.algorithms.components.is_strongly_connected(G)


def fisher_z(adj):
    np.fill_diagonal(adj, 0)
    return 0.5 * np.log((1 + adj) / (1 - adj))


def to_distance(adj):
    return 1 - np.sqrt((1 - adj) / 2)


def drop_negative(adj):
    return adj[adj >= 0]


def concat_node_feature(data):
    data.x = torch.cat([data.x, data.adj_statistics, data.raw_adj], dim=-1)
    return data

# def cv_split_group(all_indexes, n_split, group_vector, random_state=None):
#     """
#
#     :param all_indexes:
#     :param n_split:
#     :param group_vector:
#     :param random_state:
#     :return:
#     """
#     if random_state:
#         np.random.seed(random_state)
#
#     train_indexes, validation_indexes = [], []
#
#     np.random.choice(group_vector)
