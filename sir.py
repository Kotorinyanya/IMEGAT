import multiprocessing
import os
import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from networkx.generators.random_graphs import erdos_renyi_graph
import random


def init_nodes(G, seed_node):
    for n in G.nodes():
        G.nodes[n]['status'] = 'S'
    G.nodes[seed_node]['status'] = 'I'


def init_graph(adj):
    G = nx.from_numpy_array(adj)
    return G


def time_step(G, i, beta, gamma):
    if G.nodes[i]['status'] != 'I':
        return

    if random.random() < gamma:
        G.nodes[i]['status'] = 'R'

    neighbors = [n for n in G.neighbors(i) if G.nodes[n]['status'] == 'S']
    for n in neighbors:
        if random.random() < beta * G.edges[i, n]['weight']:  # weighted
            G.nodes[n]['status'] = 'I'

    if random.random() < gamma:
        G.nodes[i]['status'] = 'R'


def sir_spread(G, seed_node, beta, gamma):
    """
    params:
        seed_node: seed node
        beta, gamma: rate
    """
    init_nodes(G, seed_node)
    node_states = [G.nodes[i]['status'] for i in G.nodes()]
    while 'I' in node_states:
        infected_nodes = [node for node in G.nodes() if G.nodes[node]['status'] == 'I']
        for i in infected_nodes:
            time_step(G, i, beta, gamma)
        node_states = [G.nodes[i]['status'] for i in G.nodes()]


def count_infection(G):
    return len([1 for i in G.nodes() if G.nodes[i]['status'] != 'S'])


def sir_simulation_rank_score_at_i(G, beta, gamma, num_iter, seed_node):
    infection_counts = []
    for i in range(num_iter):
        init_nodes(G, seed_node)
        sir_spread(G, seed_node, beta=beta, gamma=gamma)
        infection_counts.append(count_infection(G))
    return np.asarray(infection_counts).mean()


def sir_score_of_adj(adj, num_iter=1000):
    """

    :param adj:
    :param num_iter:
    :return: shape (num_nodes, num_features)
    """
    adj = np.abs(adj)
    G = init_graph(adj)

    beta_c = adj.sum(0).mean() / (np.power(adj.sum(0), 2).mean() - adj.sum(0).mean())
    beta_range = np.asarray([1, 1.5, 2])
    betas = beta_c * beta_range

    gamma = 1

    all_beta_node_scores = []
    with multiprocessing.Pool(int(os.cpu_count() * 0.8)) as pool:
        for beta in betas:
            func = partial(sir_simulation_rank_score_at_i, G, beta, gamma, num_iter)
            node_scores = list(pool.map(func, list(G.nodes)))
            all_beta_node_scores.append(node_scores)

    #     node_rank = sorted(range(len(node_scores)), key=lambda k: node_scores[k])
    all_beta_node_scores = torch.tensor(all_beta_node_scores)
    all_beta_node_scores = all_beta_node_scores.t()
    return all_beta_node_scores
