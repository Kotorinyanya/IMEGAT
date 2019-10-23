import logging
import os.path as osp

import torch
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


def doubly_stochastic_normalization_d(edge_index, edge_attr, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    adj = to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes)
    adj = adj.toarray()

    tilde_adj = adj / adj.sum(axis=1)

    for u, (i, j) in enumerate(edge_index.t()):
        E_i_j = 0
        for k in range(0, num_nodes):
            E_i_j += tilde_adj[i][k] * tilde_adj[j][k] / tilde_adj[:, k].sum()
        edge_attr[u] = E_i_j

    return edge_attr


def doubly_stochastic_normalization_adj(adj):
    """

    :param adj: Numpy array
    :return:
    """
    num_nodes = adj.shape[0]

    tilde_adj = adj / adj.sum(axis=1)
    col_sums = tilde_adj.sum(axis=1)

    normed_adj = np.zeros_like(adj)
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            normed_adj[i, j] = np.sum(tilde_adj[i] * tilde_adj[j] / col_sums)

    # to symmetric
    normed_adj = np.maximum(normed_adj, normed_adj.transpose())

    return normed_adj


def positive_transform(conn_matrix):
    return 1 - np.sqrt((1 - conn_matrix) / 2)


def transform_e(adj):
    adj = positive_transform(adj)
    adj = doubly_stochastic_normalization_adj(adj)
    return adj
