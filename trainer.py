import os.path as osp
from itertools import combinations, combinations_with_replacement

import sklearn
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from boxx import timeit
from sklearn.model_selection import KFold, GroupShuffleSplit, GroupKFold, StratifiedKFold
from utils import *
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
import time
import numpy as np

from warmup_scheduler import GradualWarmupScheduler

from livelossplot import PlotLosses
import pandas as pd


# torch.autograd.set_detect_anomaly(True)


def train_val(model, optimizer, dataloader, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    # loss logging
    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_nll_loss = 0.0
    # running_domain_loss = 0.0
    epoch_label, epoch_predicted = torch.tensor([]), torch.tensor([])

    for data_list in dataloader:

        y_hat, domain_yhat, reg = model(data_list)

        device = y_hat.device

        y = torch.tensor([], dtype=data_list[0].y.dtype, device=device)
        domain_y = torch.tensor([], dtype=data_list[0].site_id.dtype, device=device)
        for data in data_list:
            y = torch.cat([y, data.y.view(-1).to(device)])
            domain_y = torch.cat([domain_y, data.site_id.view(-1).to(device)])

        loss = F.nll_loss(y_hat, y)
        domain_loss = F.nll_loss(domain_yhat, domain_y) if domain_yhat else None
        # domain_loss = -1e-7 * domain_loss
        # print(domain_loss.item())。

        total_loss = loss
        if reg is not None:
            total_loss += reg.sum()
        if domain_loss is not None:
            total_loss += domain_loss.sum()

        _, predicted = torch.max(y_hat, 1)
        label = y

        if phase == 'train':
            # print(torch.autograd.grad(y_hat.sum(), model.saved_x, retain_graph=True))
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        running_nll_loss += loss.item()
        running_total_loss += total_loss.item()
        # running_domain_loss += domain_loss.item()
        running_reg_loss += reg.sum().item()

        epoch_label = torch.cat([epoch_label, label.detach().float().view(-1).cpu()])
        epoch_predicted = torch.cat([epoch_predicted, predicted.detach().float().view(-1).cpu()])

    precision = sklearn.metrics.precision_score(epoch_label, epoch_predicted)
    recall = sklearn.metrics.recall_score(epoch_label, epoch_predicted)
    f1_score = sklearn.metrics.f1_score(epoch_label, epoch_predicted)
    accuracy = sklearn.metrics.accuracy_score(epoch_label, epoch_predicted)
    total_loss = running_total_loss / dataloader.__len__()
    nll_loss = running_nll_loss / dataloader.__len__()
    reg_loss = running_reg_loss / dataloader.__len__()

    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'total_loss': total_loss,
        'nll_loss': nll_loss,
        'reg_loss': reg_loss,
    }


def init(model_cls, use_gpu=False, dp=False, ddp=False, device_ids=None,
         seed=None, comment=''):
    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    model_name = model_cls.__name__

    if device_ids and dp:
        device = device_ids[0]
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # TensorBoard
    log_dir_base = get_model_log_dir(comment, model_name)
    print('run: ', log_dir_base)

    return log_dir_base


def init_model(model_cls, log_dir_base, fold_no, device_ids=None,
               use_gpu=False, dp=False, ddp=False,
               tb_dir='runs',
               lr=1e-3, weight_decay=1e-2):
    writer = SummaryWriter(log_dir=osp.join(tb_dir, log_dir_base))

    model = model_cls(writer)

    writer.add_text('model_summary', model.__repr__())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    # scheduler_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5)
    # scheduler = scheduler_reduce
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if dp and use_gpu:
        model = model.cuda() if device_ids is None else model.to(device_ids[0])
        model = DataParallel(model, device_ids=device_ids)
    elif use_gpu:
        model = model.to(device_ids[0])

    device_count = torch.cuda.device_count() if dp else 1
    device_count = len(device_ids) if (device_ids is not None and dp) else device_count

    return model, optimizer, writer, device_count


def cv_prepare_dataloader(dataset, batch_size, fold_no, n_splits=10, fold_seed=None,
                          ):
    iter = multi_site_cv_split(dataset.data.y, dataset.data.site_id, dataset.data.subject_id, n_splits,
                               random_state=fold_seed, shuffle=True if fold_seed else False)
    train_idx, val_idx = list(iter)[fold_no]

    train_dataset, val_dataset = norm_train_val(dataset, train_idx, val_idx, num_nodes=360)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  collate_fn=lambda data_list: data_list)
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                batch_size=batch_size,
                                collate_fn=lambda data_list: data_list)
    return train_dataloader, val_dataloader


def train_main(model_cls, dataset, lr=1e-3,
               weight_decay=1e-2, num_epochs=200, n_splits=10,
               use_gpu=True, dp=False, ddp=False,
               comment='', batch_size=1, fold_no=1,
               device_ids=None, patience=50, seed=None, fold_seed=None,
               save_model=False):
    # save configuration
    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    log_dir_base = init(model_cls, use_gpu, dp, ddp, device_ids, seed, comment)

    pass


def train_cross_site(model_cls, dataset, sites, lr=1e-3,
                     weight_decay=1e-2, num_epochs=200, n_splits=5,
                     use_gpu=True, dp=False, ddp=False,
                     comment='', batch_size=1, fold_no=1,
                     tb_dir='runs', model_save_dir='saved_models', res_save_dir='res',
                     device_ids=None, patience=50, seed=None, fold_seed=None,
                     save_model=False):
    # save configuration
    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    log_dir_base = init(model_cls, use_gpu, dp, ddp, device_ids, seed, comment)
    log_dir_base += str(fold_no)

    res_save_dir = osp.join(res_save_dir, log_dir_base)

    # sites = ["NYU", "UM_1"]
    comb = combinations_with_replacement(sites, 2)

    train_res_df = pd.DataFrame()
    val_res_df = pd.DataFrame()

    for train_site in sites:
        model, optimizer, writer, k = init_model(model_cls, log_dir_base, fold_no, device_ids, use_gpu, dp, ddp,
                                                 tb_dir, lr, weight_decay)
        # train in one, val in all
        train_dataset = dataset.filter_by_site([train_site])
        val_datasets = [dataset.filter_by_site([val_site])
                        for val_site in sites]

        train_dataloader = cv_prepare_dataloader(train_dataset, batch_size * k, fold_no, n_splits, fold_seed)[0]
        val_dataloaders = [cv_prepare_dataloader(val_dataset, batch_size * k, fold_no, n_splits, fold_seed)[1]
                           for val_dataset in val_datasets]

        for epoch in range(num_epochs):

            # train
            train_res = train_val(model, optimizer, train_dataloader, 'train')
            train_res['epoch'] = epoch
            train_res['site_train'] = train_site
            train_res['site_res'] = train_site
            train_res_df = train_res_df.append(train_res, ignore_index=True)
            my_save(train_res_df, osp.join(res_save_dir, 'train_res_df'))

            # val
            for val_site, val_dataloader in zip(sites, val_dataloaders):
                val_res = train_val(model, optimizer, val_dataloader, 'val')
                val_res['epoch'] = epoch
                val_res['site_train'] = train_site
                val_res['site_res'] = val_site
                val_res_df = val_res_df.append(val_res, ignore_index=True)
            my_save(val_res_df, osp.join(res_save_dir, 'val_res_df'))


def train_single_site(model_cls, dataset, sites, lr=1e-3,
                      weight_decay=1e-2, num_epochs=200, n_splits=5,
                      use_gpu=True, dp=False, ddp=False,
                      comment='', batch_size=1, fold_no=1,
                      tb_dir='runs', model_save_dir='saved_models', res_save_dir='res',
                      device_ids=None, seed=None, fold_seed=None,
                      save_model=False):
    # save configuration
    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    log_dir_base = init(model_cls, use_gpu, dp, ddp, device_ids, seed, comment)
    log_dir_base += str(fold_no)

    res_save_dir = osp.join(res_save_dir, log_dir_base)

    res_df = pd.DataFrame()

    model, optimizer, writer, k = init_model(model_cls, log_dir_base, fold_no, device_ids, use_gpu, dp, ddp,
                                             tb_dir, lr, weight_decay)

    dataset = dataset.filter_by_site(sites)
    assert len(sites) == 1
    site = sites[0]

    train_dataloader, val_dataloader = cv_prepare_dataloader(dataset, batch_size * k, fold_no, n_splits, fold_seed)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
            else:
                dataloader = val_dataloader
            res = train_val(model, optimizer, dataloader, phase)
            res['epoch'] = epoch
            res['site'] = site
            res['phase'] = phase
            res_df = res_df.append(res, ignore_index=True)

            my_save(res_df, osp.join(res_save_dir, 'res_df'))

            writer.add_scalars('nll_loss',
                               {'{}_nll_loss'.format(phase): res['nll_loss']},
                               epoch)
            writer.add_scalars('accuracy',
                               {'{}_accuracy'.format(phase): res['accuracy']},
                               epoch)
            if res['reg_loss'] is not None:
                writer.add_scalars('reg_loss'.format(phase),
                                   {'{}_reg_loss'.format(phase): res['reg_loss']},
                                   epoch)


def train_cross_validation(model_cls, dataset, dropout=0.0, lr=1e-3,
                           weight_decay=1e-2, num_epochs=200, n_splits=10,
                           use_gpu=True, dp=False, ddp=False,
                           comment='', tb_service_loc='192.168.192.57:6007', batch_size=1,
                           num_workers=0, pin_memory=False, cuda_device=None,
                           tb_dir='runs', model_save_dir='saved_models', res_save_dir='res',
                           fold_no=None, saved_model_path=None,
                           device_ids=None, patience=20, seed=None, fold_seed=None,
                           save_model=False, is_reg=True, live_loss=True, domain_cls=True, final_cls=True):
    """
    :type fold_seed: int
    :param live_loss: bool
    :param is_reg: bool
    :param save_model: bool
    :param seed:
    :param patience: for early stopping
    :param device_ids: for ddp
    :param saved_model_path:
    :param fold_no: int
    :param ddp_port: str
    :param ddp: DDP
    :param cuda_device: list of int
    :param pin_memory: bool, DataLoader args
    :param num_workers: int, DataLoader args
    :param model_cls: pytorch Module cls
    :param dataset: instance
    :param dropout: float
    :param lr: float
    :param weight_decay:
    :param num_epochs:
    :param n_splits: number of kFolds
    :param use_gpu: bool
    :param dp: bool
    :param comment: comment in the logs, to filter runs in tensorboard
    :param tb_service_loc: tensorboard service location
    :param batch_size: Dataset args not DataLoader
    :return:
    """
    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    model_name = model_cls.__name__

    if not cuda_device:
        if device_ids and dp:
            device = device_ids[0]
        else:
            device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    else:
        device = cuda_device

    device_count = torch.cuda.device_count() if dp else 1
    device_count = len(device_ids) if (device_ids is not None and dp) else device_count

    batch_size = batch_size * device_count

    # TensorBoard
    log_dir_base = get_model_log_dir(comment, model_name)
    if tb_service_loc is not None:
        print("TensorBoard available at http://{1}/#scalars&regexInput={0}".format(
            log_dir_base, tb_service_loc))
    else:
        print("Please set up TensorBoard")

    # model
    criterion = nn.NLLLoss()

    print("Training {0} {1} models for cross validation...".format(n_splits, model_name))
    # 1
    # folds, fold = KFold(n_splits=n_splits, shuffle=False, random_state=seed), 0
    # 2
    # folds = GroupKFold(n_splits=n_splits)
    # iter = folds.split(np.zeros(len(dataset)), groups=dataset.data.site_id)
    # 4
    # folds = StratifiedKFold(n_splits=n_splits, random_state=fold_seed, shuffle=True if fold_seed else False)
    # iter = folds.split(np.zeros(len(dataset)), dataset.data.y.numpy(), groups=dataset.data.subject_id)
    # 5
    fold = 0
    iter = multi_site_cv_split(dataset.data.y, dataset.data.site_id, dataset.data.subject_id, n_splits,
                               random_state=fold_seed, shuffle=True if fold_seed else False)

    for train_idx, val_idx in tqdm_notebook(iter, desc='CV', leave=False):
        fold += 1
        liveloss = PlotLosses() if live_loss else None

        # for a specific fold
        if fold_no is not None:
            if fold != fold_no:
                continue

        writer = SummaryWriter(log_dir=osp.join('runs', log_dir_base + str(fold)))
        model_save_dir = osp.join('saved_models', log_dir_base + str(fold))

        print("creating dataloader tor fold {}".format(fold))

        train_dataset, val_dataset = norm_train_val(dataset, train_idx, val_idx)

        model = model_cls(writer)

        train_dataloader = DataLoader(train_dataset,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      collate_fn=lambda data_list: data_list,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    collate_fn=lambda data_list: data_list,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)

        if fold == 1 or fold_no is not None:
            print(model)
            writer.add_text('model_summary', model.__repr__())
            writer.add_text('training_args', str(saved_args))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # scheduler_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5)
        # scheduler = scheduler_reduce
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if dp and use_gpu:
            model = model.cuda() if device_ids is None else model.to(device_ids[0])
            model = DataParallel(model, device_ids=device_ids)
        elif use_gpu:
            model = model.to(device)

        if saved_model_path is not None:
            model.load_state_dict(torch.load(saved_model_path))

        best_map, patience_counter, best_score = 0.0, 0, np.inf
        for epoch in tqdm_notebook(range(1, num_epochs + 1), desc='Epoch', leave=False):
            logs = {}

            # scheduler.step(epoch=epoch, metrics=best_score)

            for phase in ['train', 'validation']:

                if phase == 'train':
                    model.train()
                    dataloader = train_dataloader
                else:
                    model.eval()
                    dataloader = val_dataloader

                # Logging
                running_total_loss = 0.0
                running_corrects = 0
                running_reg_loss = 0.0
                running_nll_loss = 0.0
                epoch_yhat_0, epoch_yhat_1 = torch.tensor([]), torch.tensor([])
                epoch_label, epoch_predicted = torch.tensor([]), torch.tensor([])

                logging_hist = True if phase == 'train' else False  # once per epoch
                for data_list in tqdm_notebook(dataloader, desc=phase, leave=False):

                    # TODO: check devices
                    if dp:
                        data_list = to_cuda(data_list, (device_ids[0] if device_ids is not None else 'cuda'))

                    y_hat, domain_yhat, reg = model(data_list)

                    y = torch.tensor([], dtype=dataset.data.y.dtype, device=device)
                    domain_y = torch.tensor([], dtype=dataset.data.site_id.dtype, device=device)
                    for data in data_list:
                        y = torch.cat([y, data.y.view(-1).to(device)])
                        domain_y = torch.cat([domain_y, data.site_id.view(-1).to(device)])

                    loss = criterion(y_hat, y)
                    domain_loss = criterion(domain_yhat, domain_y)
                    # domain_loss = -1e-7 * domain_loss
                    # print(domain_loss.item())
                    if domain_cls:
                        total_loss = domain_loss
                        _, predicted = torch.max(domain_yhat, 1)
                        label = domain_y
                    if final_cls:
                        total_loss = loss
                        _, predicted = torch.max(y_hat, 1)
                        label = y
                    if domain_cls and final_cls:
                        total_loss = (loss + domain_loss).sum()
                        _, predicted = torch.max(y_hat, 1)
                        label = y

                    if is_reg:
                        total_loss += reg.sum()

                    if phase == 'train':
                        # print(torch.autograd.grad(y_hat.sum(), model.saved_x, retain_graph=True))
                        optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()

                    running_nll_loss += loss.item()
                    running_total_loss += total_loss.item()
                    running_reg_loss += reg.sum().item()
                    running_corrects += (predicted == label).sum().item()

                    epoch_yhat_0 = torch.cat([epoch_yhat_0, y_hat[:, 0].detach().view(-1).cpu()])
                    epoch_yhat_1 = torch.cat([epoch_yhat_1, y_hat[:, 1].detach().view(-1).cpu()])
                    epoch_label = torch.cat([epoch_label, label.detach().float().view(-1).cpu()])
                    epoch_predicted = torch.cat([epoch_predicted, predicted.detach().float().view(-1).cpu()])

                # precision = sklearn.metrics.precision_score(epoch_label, epoch_predicted, average='micro')
                # recall = sklearn.metrics.recall_score(epoch_label, epoch_predicted, average='micro')
                # f1_score = sklearn.metrics.f1_score(epoch_label, epoch_predicted, average='micro')
                accuracy = sklearn.metrics.accuracy_score(epoch_label, epoch_predicted)
                epoch_total_loss = running_total_loss / dataloader.__len__()
                epoch_nll_loss = running_nll_loss / dataloader.__len__()
                epoch_reg_loss = running_reg_loss / dataloader.__len__()

                # print('epoch {} {}_nll_loss: {}'.format(epoch, phase, epoch_nll_loss))
                writer.add_scalars('nll_loss',
                                   {'{}_nll_loss'.format(phase): epoch_nll_loss},
                                   epoch)
                writer.add_scalars('accuracy',
                                   {'{}_accuracy'.format(phase): accuracy},
                                   epoch)
                # writer.add_scalars('{}_APRF'.format(phase),
                #                    {
                #                        'accuracy': accuracy,
                #                        'precision': precision,
                #                        'recall': recall,
                #                        'f1_score': f1_score
                #                    },
                #                    epoch)
                if epoch_reg_loss != 0:
                    writer.add_scalars('reg_loss'.format(phase),
                                       {'{}_reg_loss'.format(phase): epoch_reg_loss},
                                       epoch)
                # print(epoch_reg_loss)
                # writer.add_histogram('hist/{}_yhat_0'.format(phase),
                #                      epoch_yhat_0,
                #                      epoch)
                # writer.add_histogram('hist/{}_yhat_1'.format(phase),
                #                      epoch_yhat_1,
                #                      epoch)

                # Save Model & Early Stopping
                if phase == 'validation':
                    model_save_path = model_save_dir + '-{}-{}-{:.3f}-{:.3f}'.format(
                        model_name, epoch, accuracy, epoch_nll_loss)
                    # best score
                    if accuracy > best_map:
                        best_map = accuracy
                        model_save_path = model_save_path + '-best'

                    score = epoch_nll_loss
                    if score < best_score:
                        patience_counter = 0
                        best_score = score
                    else:
                        patience_counter += 1

                    # skip first 10 epoch
                    # best_score = best_score if epoch > 10 else -np.inf

                    if save_model:
                        for th, pfix in zip([0.8, 0.75, 0.7, 0.5, 0.0],
                                            ['-perfect', '-great', '-good', '-bad', '-miss']):
                            if accuracy >= th:
                                model_save_path += pfix
                                break

                        torch.save(model.state_dict(), model_save_path)

                    writer.add_scalars('best_val_accuracy',
                                       {'{}_accuracy'.format(phase): best_map},
                                       epoch)
                    writer.add_scalars('best_nll_loss',
                                       {'{}_nll_loss'.format(phase): best_score},
                                       epoch)

                    writer.add_scalars('learning_rate',
                                       {'learning_rate': scheduler.optimizer.param_groups[0]['lr']},
                                       epoch)

                    if patience_counter >= patience:
                        print("Stopped at epoch {}".format(epoch))
                        return

                if live_loss:
                    prefix = ''
                    if phase == 'validation':
                        prefix = 'val_'

                    logs[prefix + 'log loss'] = epoch_nll_loss
                    logs[prefix + 'accuracy'] = accuracy
            if live_loss:
                liveloss.update(logs)
                liveloss.draw()

    print("Done !")


if __name__ == "__main__":
    from dataset import ABIDE
    from model import *

    dataset = ABIDE(root='datasets/ALL')
    sites = ['NYU', 'USM', 'UM_1', 'UCLA_1']
    dataset = dataset.filter_by_site(sites)
    model = Net
    # train_cross_validation(model, dataset, comment='test', batch_size=8, patience=200,
    #                        num_epochs=200, dropout=0.5, lr=3e-4, weight_decay=0.01, n_splits=5,
    #                        use_gpu=True, dp=False, ddp=False, fold_no=1,
    #                        device_ids=[4], cuda_device=1, fold_seed=None)
    # train_cross_site(model, dataset, sites, device_ids=[4], comment='debug')
    train_single_site(model, dataset, ['NYU'], device_ids=[4], comment='debug')
