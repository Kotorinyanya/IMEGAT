import logging
import os.path as osp

import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
import socket
from datetime import datetime


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


def z_score_norm(tensor):
    """
    Normalize a tensor with mean and standard deviation.
    Args:
        tensor (Tensor): Tensor image of size [num_nodes, num_node_features] to be normalized.

    Returns:
        Tensor: Normalized tensor.
    """
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std == 0] = 1e-6
    normed_tensor = (tensor - mean) / std
    return normed_tensor


def z_score_norm_data(data):
    data.x = z_score_norm(data.x)
    return data


def gaussian_fit(data):
    data.x = data.x.normal_(mean=0, std=1)
    return data


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


def entropy(src, edge_index, num_nodes=None):
    EPS = 1e-32
    index, _ = edge_index
    num_nodes = maybe_num_nodes(index, num_nodes)

    # out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = -src * torch.log(src + EPS)
    out = scatter_add(out, index, dim=0, dim_size=num_nodes)

    return out


def from_2d_tensor_adj(adj):
    """
    maintain gradients
    Args:
        A : Tensor
    """
    edge_index = adj.nonzero().t().detach()
    row, col = edge_index
    edge_weight = adj[row, col]

    assert len(row) == len(edge_weight)
    return edge_index, edge_weight


def nan_or_inf(x):
    return torch.isnan(x).any() or x.eq(float('inf')).any() or x.eq(float('-inf')).any()
