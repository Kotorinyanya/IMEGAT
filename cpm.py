import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupShuffleSplit, GroupKFold, StratifiedKFold
import sklearn
from multiprocessing import Pool
import os

from sklearn import linear_model
import numpy as np
import scipy.stats as stat


class LogisticReg():
    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance
    in an attribute self.model, and pvalues, z scores and estimated
    errors for each coefficient in

    self.z_scores
    self.p_values
    self.sigma_estimates

    as well as the negative hessian of the log Likelihood (Fisher information)

    self.F_ij
    """

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)  ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates  # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  ### two tailed test for p-values

        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij


def get_edge_and_y(dataset):
    edge_list = []
    labels = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        ea, ei = data.edge_attr, data.edge_index
        adj = to_scipy_sparse_matrix(ei, ea).toarray()
        np.fill_diagonal(adj, 1)
        edge_list.append(adj.reshape(-1))
        labels.append(data.y.numpy())
    edge_list = np.stack(edge_list, -1)
    labels = np.stack(labels, -1)
    return edge_list, labels


def p_edge(edge_list, labels):
    p_values = []
    for i in tqdm(range(edge_list.shape[0])):
        model = LogisticReg(solver='liblinear')
        model.fit(edge_list[i, :].reshape(-1, 1), labels.reshape(-1))
        p_values += model.p_values
    p_values = np.asarray(p_values)
    return p_values


def train_cpm(dataset, mask):
    edge_list, labels = get_edge_and_y(dataset)

    new_x = (edge_list * mask).sum(0)

    model = LogisticReg(solver='liblinear')
    model.fit(new_x.reshape(-1, 1), labels.reshape(-1))
    return model


def val_cpm(dataset, mask, model):
    edge_list, labels = get_edge_and_y(dataset)
    new_x = (edge_list * mask).sum(0)
    prediction = model.model.predict_proba(new_x.reshape(-1, 1))
    return prediction


def get_y(arr):
    y = np.zeros((arr.shape[0], arr.max() + 1))
    y[arr == 1, 1] = 1
    return y


def train_val(train_dataset, test_dataset, p_values, th, model='cpm'):
    edge_list, labels = get_edge_and_y(train_dataset)
    mask = (p_values <= th).reshape(-1, 1)

    new_x = (edge_list * mask).sum(0)

    if model == 'cpm':
        model = train_cpm(train_dataset, mask)
        val_predictions = val_cpm(test_dataset, mask, model)
        train_predictions = model.model.predict_proba(new_x.reshape(-1, 1))

    if model == 'mlp':
        pass

    truth_train_y = train_dataset.data.y
    truth_val_y = test_dataset.data.y

    val_acc = sklearn.metrics.accuracy_score(truth_val_y, val_predictions.argmax(1))
    train_acc = sklearn.metrics.accuracy_score(truth_train_y, train_predictions.argmax(1))
    val_auc = sklearn.metrics.roc_auc_score(truth_val_y, val_predictions[:, 1])
    train_auc = sklearn.metrics.roc_auc_score(truth_train_y, train_predictions[:, 1])
    return train_acc, val_acc, train_auc, val_auc


def get_p_values(dataset, iter):
    p_values = []
    for train_idx, test_idx in tqdm(iter,
                                    desc='models', leave=False):
        train_dataset = dataset.__indexing__(train_idx)
        edge_list, labels = get_edge_and_y(train_dataset)
        p_value = p_edge(edge_list, labels)
        p_values.append(p_value)
    return p_values


def cv(dataset, ths):
    # folds = GroupShuffleSplit(n_splits=4, train_size=int(len(dataset)/2), test_size=int(len(dataset)/2))
    folds, fold = StratifiedKFold(n_splits=10), 0
    # p_values = get_p_values(dataset, iter)
    import torch
    p_values = torch.load('p')
    rep_dict = {}
    for th in ths:
        fold = 0
        train_accs, val_accs, train_aucs, val_aucs = [], [], [], []
        iter = folds.split(np.zeros(len(dataset)), dataset.data.y.numpy())
        for train_idx, test_idx in tqdm(iter,
                                        desc='models', leave=False):
            train_dataset = dataset.__indexing__(train_idx)
            test_dataset = dataset.__indexing__(test_idx)
            p_value_arr = p_values[fold]
            fold += 1

            train_acc, val_acc, train_auc, val_auc = train_val(train_dataset, test_dataset, p_value_arr, th)
            #         print(train_acc, val_acc)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
        print(train_accs, val_accs, train_aucs, val_aucs)
        train_accs = np.asarray(train_accs)
        val_accs = np.asarray(val_accs)
        train_aucs = np.asarray(train_aucs)
        val_aucs = np.asarray(val_aucs)
        rep_train_acc = (train_accs.mean(), train_accs.std())
        rep_val_acc = (val_accs.mean(), val_accs.std())
        rep_train_auc = (train_aucs.mean(), train_auc.std())
        rep_val_auc = (val_aucs.mean(), val_aucs.std())
        rep_dict[th] = (rep_train_acc, rep_val_acc, rep_train_auc, rep_val_auc)
        print(rep_dict)

    return rep_dict


if __name__ == '__main__':
    from dataset import ABIDE

    dataset = ABIDE(root='datasets/ALL')

    ths = []
    for i in range(10):
        ths.append(0.01 / (2 ** i))

    rep_dict = cv(dataset, ths)
    print(rep_dict)

    """
    {0.01: ((0.6117754401030485, 0.048469536474122166),
  (0.5847058823529412, 0.051091884119078744)),
 0.005: ((0.6358630313439245, 0.05061641611701489),
  (0.6052941176470588, 0.06270709633523026)),
 0.0025: ((0.6461195792185488, 0.04417452657347166),
  (0.6052941176470588, 0.05915787334952765)),
 0.00125: ((0.6622048089308716, 0.04227550411998894),
  (0.6111764705882352, 0.06358385466763135)),
 0.000625: ((0.6754186346071276, 0.02461789085582714),
  (0.616890756302521, 0.056398012586803645)),
 0.0003125: ((0.6783437097466724, 0.022988232715158614),
  (0.6341176470588235, 0.07355528956395571)),
 0.00015625: ((0.6863782739373121, 0.018080188062004038),
  (0.6312605042016808, 0.066790918272222)),
 7.8125e-05: ((0.689303349076857, 0.013808735310562618),
  (0.6371428571428572, 0.05368597080820969)),
 3.90625e-05: ((0.6900332760841563, 0.01447388960459478),
  (0.628235294117647, 0.056867477828738584)),
 1.953125e-05: ((0.6944182052382997, 0.012561221566370427),
  (0.6341176470588235, 0.0584872742199082))}
    """
