#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Meir Yossef Levi
@Contact: me.levi@campus.technion.ac.il
@File: main
@Time: 23/04/10
"""

from __future__ import print_function
import os
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from models.dgcnn.model_dgcnn import DGCNN, DGCNN_v1, DGCNN_v2, DGCNN_v3, DGCNN_v5
from models.curvenet.curvenet_cls import CurveNet
from models.gdanet.GDANet_cls import GDANET
from models.pct.model_pct import Pct, RPC

# placeholder to future researches to implant and examine their classification networks
from models.custom.custom_model import custom_model

from epic_util import *
from augmentation.rsmix import provider, rsmix_provider

# for test
from modelnetc_utils import eval_corrupt_wrapper, ModelNetC
import random

import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics


def set_train(args, model_random, model_patches, model_curves):
    if args.train_random:
        model_random.train()
    if args.train_patches:
        model_patches.train()
    if args.train_curves:
        model_curves.train()


def set_eval(args, model_random, model_patches, model_curves):
    if args.train_random:
        model_random.eval()
    if args.train_patches:
        model_patches.eval()
    if args.train_curves:
        model_curves.eval()


def set_scheduler(args, scheduler_random, scheduler_patches, scheduler_curves):
    if args.train_random:
        scheduler_random.step()
    if args.train_patches:
        scheduler_patches.step()
    if args.train_curves:
        scheduler_curves.step()


def set_zero_grad(args, opt_random, opt_patches, opt_curves):
    if args.train_random:
        opt_random.zero_grad()
    if args.train_patches:
        opt_patches.zero_grad()
    if args.train_curves:
        opt_curves.zero_grad()


def extract_ppcs(args, data, anchor):
    if args.train_random:
        random_ppc = extract_random(data, args.nr)
    else:
        random_ppc = None
    if args.train_patches:
        patches_ppc = extract_patches(data, anchor, args.np)
    else:
        patches_ppc = None
    if args.train_curves:
        curves_ppc = extract_curves(data, anchor, args.m, args.nc)
    else:
        curves_ppc = None
    return random_ppc, patches_ppc, curves_ppc


def apply_models(args, model_random, model_patches, model_curves, data_random, data_patches, data_curves):
    if args.train_random:
        logits_random = model_random(data_random)
    else:
        logits_random = None
    if args.train_patches:
        logits_patches = model_patches(data_patches)
    else:
        logits_patches = None
    if args.train_curves:
        logits_curves = model_curves(data_curves)
    else:
        logits_curves = None
    return logits_random, logits_patches, logits_curves


def calculate_losses(args, logits_random, logits_patches, logits_curves, label, criterion, rsmix, lam, label_b):
    batch_size = logits_patches.shape[0]
    if args.train_random:
        if not rsmix:
            loss_random = criterion(logits_random, label)
        else:
            loss = 0
            for i in range(batch_size):
                loss_tmp = criterion(logits_random[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - lam[i]) \
                           + criterion(logits_random[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                loss += loss_tmp
            loss_random = loss / batch_size

    else:
        loss_random = None
    if args.train_patches:
        if not rsmix:
            loss_patches = criterion(logits_patches, label)
        else:
            loss = 0
            for i in range(batch_size):
                loss_tmp = criterion(logits_patches[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - lam[i]) \
                           + criterion(logits_patches[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                loss += loss_tmp
            loss_patches = loss / batch_size
    else:
        loss_patches = None
    if args.train_curves:
        if not rsmix:
            loss_curves = criterion(logits_curves, label)
        else:
            loss = 0
            for i in range(batch_size):
                loss_tmp = criterion(logits_curves[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - lam[i]) \
                           + criterion(logits_curves[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                loss += loss_tmp
            loss_curves = loss / batch_size
    else:
        loss_curves = None
    return loss_random, loss_patches, loss_curves


def backward_loss(args, loss_random, loss_patches, loss_curves):
    if args.train_random:
        loss_random.backward()
    if args.train_patches:
        loss_patches.backward()
    if args.train_curves:
        loss_curves.backward()


def step_opts(args, opt_random, opt_patches, opt_curves):
    if args.train_random:
        opt_random.step()
    if args.train_patches:
        opt_patches.step()
    if args.train_curves:
        opt_curves.step()


def m_concatenate(args, train_random, train_patches, train_curves):
    if args.train_random:
        train_random = np.concatenate(train_random)
    if args.train_patches:
        train_patches = np.concatenate(train_patches)
    if args.train_curves:
        train_curves = np.concatenate(train_curves)
    return train_random, train_patches, train_curves


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    os.system('cp epic_util.py checkpoints' + '/' + args.exp_name + '/' + 'epic_util.py.backup')


def train(args, io):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train_loader = DataLoader(ModelNet40(args=args, partition='train'), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(args=args, partition='test'),
                             num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    model_random, model_patches, model_curves = None, None, None
    opt_random, opt_patches, opt_curves = None, None, None
    scheduler_random, scheduler_patches, scheduler_curves = None, None, None
    if args.train_random:
        model_random = load_model(args, device)
        total_params_random = sum(param.numel() for param in model_random.parameters())
        print(f'random params: {total_params_random}')
        opt_random = optim.SGD(model_random.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler_random = CosineAnnealingLR(opt_random, args.epochs)
    if args.train_patches:
        model_patches = load_model(args, device)
        total_params_patches = sum(param.numel() for param in model_patches.parameters())
        print(f'patches params: {total_params_patches}')
        opt_patches = optim.SGD(model_patches.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler_patches = CosineAnnealingLR(opt_patches, args.epochs)
    if args.train_curves:
        model_curves = load_model(args, device)
        total_params_curves = sum(param.numel() for param in model_curves.parameters())
        print(f'patches curves: {total_params_curves}')
        opt_curves = optim.SGD(model_curves.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler_curves = CosineAnnealingLR(opt_curves, args.epochs)
    if not args.train_random and not args.train_patches and not args.train_curves:
        print("you must train random, curves or patches")
        exit(-1)

    criterion = cal_loss

    best_test_acc_random = 0
    best_test_acc_patches = 0
    best_test_acc_curves = 0
    for epoch in range(args.epochs):
        set_scheduler(args, scheduler_random, scheduler_patches, scheduler_curves)
        ####################
        # Train
        ####################
        train_loss_random = 0.0
        train_loss_patches = 0.0
        train_loss_curves = 0.0
        count = 0.0
        set_train(args, model_random, model_patches, model_curves)
        train_pred_random = []
        train_pred_patches = []
        train_pred_curves = []
        train_true = []
        for m_iter, (data, label) in enumerate(train_loader):
            rsmix = False
            lam = False
            label_b = False
            if args.use_wolfmix:
                # RSMIX
                data, label, label_b, lam, rsmix = apply_rsmix(args, data, device, label, label_b, lam, rsmix)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size, _, num_points = data.shape
            anchors = torch.randperm(num_points).to(device)
            for cur_ppc in range(4):
                set_zero_grad(args, opt_random, opt_patches, opt_curves)
                anchor = anchors[cur_ppc].reshape(-1, 1).repeat(batch_size, 1)
                ppc_random, ppc_patches, ppc_curves = extract_ppcs(args, data, anchor)
                logits_random, logits_patches, logits_curves = apply_models(args, model_random, model_patches,
                                                                            model_curves, ppc_random, ppc_patches,
                                                                            ppc_curves)
                loss_random, loss_patches, loss_curves = calculate_losses(args, logits_random, logits_patches,
                                                                          logits_curves, label, criterion, rsmix, lam,
                                                                          label_b)
                backward_loss(args, loss_random, loss_patches, loss_curves)
                step_opts(args, opt_random, opt_patches, opt_curves)
                count += batch_size
                train_true.append(label.cpu().numpy())
                if args.train_random:
                    preds_random = logits_random.max(dim=1)[1]
                    train_loss_random += loss_random.item() * batch_size
                    train_pred_random.append(preds_random.detach().cpu().numpy())
                if args.train_patches:
                    preds_patches = logits_patches.max(dim=1)[1]
                    train_loss_patches += loss_patches.item() * batch_size
                    train_pred_patches.append(preds_patches.detach().cpu().numpy())
                if args.train_curves:
                    preds_curves = logits_curves.max(dim=1)[1]
                    train_loss_curves += loss_curves.item() * batch_size
                    train_pred_curves.append(preds_curves.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred_random, train_pred_patches, train_pred_curves = m_concatenate(args, train_pred_random,
                                                                                 train_pred_patches, train_pred_curves)

        train_acc_random = metrics.accuracy_score(train_true, train_pred_random)
        train_acc_patches = metrics.accuracy_score(train_true, train_pred_patches)
        train_acc_curves = metrics.accuracy_score(train_true, train_pred_curves)
        print_results(args, count, epoch, io, train_loss_curves,
                      train_loss_patches, train_loss_random,
                      train_pred_curves, train_pred_patches,
                      train_pred_random, train_true, train_acc_random, train_acc_patches, train_acc_curves, True)

        with torch.no_grad():
            ####################
            # Test
            ####################
            test_loss_random = 0.0
            test_loss_patches = 0.0
            test_loss_curves = 0.0
            count = 0.0
            set_eval(args, model_random, model_patches, model_curves)
            test_pred_random = []
            test_pred_patches = []
            test_pred_curves = []
            test_true = []
            for m_iter, (data, label) in enumerate(test_loader):
                torch.cuda.empty_cache()
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size, _, num_points = data.shape
                anchors = torch.randperm(num_points).to(device).reshape(-1, 1)
                for cur_ppc in range(8):
                    anchor = anchors[cur_ppc].reshape(-1, 1).repeat(batch_size, 1)
                    ppc_random, ppc_patches, ppc_curves = extract_ppcs(args, data, anchor)
                    logits_random, logits_patches, logits_curves = apply_models(args, model_random, model_patches,
                                                                                model_curves, ppc_random, ppc_patches,
                                                                                ppc_curves)
                    loss_random, loss_patches, loss_curves = calculate_losses(args, logits_random, logits_patches,
                                                                              logits_curves, label, criterion, False,
                                                                              None, None)
                    count += batch_size
                    if args.train_random:
                        preds_random = logits_random.max(dim=1)[1]
                        test_loss_random += loss_random.item() * batch_size
                        test_pred_random.append(preds_random.detach().cpu().numpy())
                    if args.train_patches:
                        preds_patches = logits_patches.max(dim=1)[1]
                        test_loss_patches += loss_patches.item() * batch_size
                        test_pred_patches.append(preds_patches.detach().cpu().numpy())
                    if args.train_curves:
                        preds_curves = logits_curves.max(dim=1)[1]
                        test_loss_curves += loss_curves.item() * batch_size
                        test_pred_curves.append(preds_curves.detach().cpu().numpy())
                    test_true.append(label.cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred_random, test_pred_patches, test_pred_curves = m_concatenate(args, test_pred_random,
                                                                                  test_pred_patches, test_pred_curves)

            test_acc_random = metrics.accuracy_score(test_true, test_pred_random)
            test_acc_patches = metrics.accuracy_score(test_true, test_pred_patches)
            test_acc_curves = metrics.accuracy_score(test_true, test_pred_curves)
            print_results(args, count, epoch, io, test_loss_curves,
                          test_loss_patches, test_loss_random,
                          test_pred_curves, test_pred_patches,
                          test_pred_random, test_true, test_acc_random, test_acc_patches, test_acc_curves, False)
            if args.train_random:
                model_path_random = os.path.join('checkpoints', args.exp_name, 'models',
                                                 f'{args.model}_random{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_random == "" else args.model_path_random
                if test_acc_random >= best_test_acc_random:
                    best_test_acc_random = test_acc_random
                    torch.save(model_random.state_dict(), model_path_random)
            if args.train_patches:
                model_path_patches = os.path.join('checkpoints', args.exp_name, 'models',
                                                  f'{args.model}_patches{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_patches == "" else args.model_path_patches
                if test_acc_patches >= best_test_acc_patches:
                    best_test_acc_patches = test_acc_patches
                    torch.save(model_patches.state_dict(), model_path_patches)
            if args.train_curves:
                model_path_curves = os.path.join('checkpoints', args.exp_name, 'models',
                                                 f'{args.model}_curves{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_curves == "" else args.model_path_curves
                if test_acc_curves >= best_test_acc_curves:
                    best_test_acc_curves = test_acc_curves
                    torch.save(model_curves.state_dict(), model_path_curves)
            io.cprint(f'RANDOM: best test: {best_test_acc_random}')
            io.cprint(f'PATCHES: best test: {best_test_acc_patches}')
            io.cprint(f'CURVES: best test: {best_test_acc_curves}')


def load_model(args, device):
    model = None
    if args.model == "gdanet":
        model = GDANET().to(device)
    elif args.model == "curvenet":
        model = CurveNet().to(device)
    elif args.model == "pct":
        model = Pct(args).to(device)
        args.test_batch_size = 50
    elif args.model == "rpc":
        model = RPC(args).to(device)
    elif args.model == "dgcnn":
        model = DGCNN(args).to(device)
    elif args.model == "dgcnn_v1":
        model = DGCNN_v1(args).to(device)
    elif args.model == 'dgcnn_v2':
        model = DGCNN_v2(args).to(device)
    elif args.model == 'dgcnn_v3':
        model = DGCNN_v3(args).to(device)
    elif args.model == 'dgcnn_v5':
        model = DGCNN_v5(args).to(device)
    # placeholder to future researches to implant and examine their classification networks 
    elif args.model == 'custom_model':
        model = custom_model(args).to(device)
    return model


def print_results(args, count, epoch, io, loss_curves, loss_patches, loss_random, pred_curves,
                  pred_patches, pred_random, true_labels, acc_random, acc_patches, acc_curves, is_train):
    outstr = ''
    train_or_test = "Train" if is_train else "Test"
    if args.train_random:
        loss_random = loss_random * 1.0 / count
        avg_acc_random = metrics.balanced_accuracy_score(true_labels, pred_random)
        outstr += f'\nRANDOM: {train_or_test} {epoch}, loss: {loss_random:.6f}, {train_or_test} acc: {acc_random:.6f},' \
                  f'  {train_or_test} avg acc: {avg_acc_random:.6f}'
    if args.train_patches:
        loss_patches = loss_patches * 1.0 / count
        avg_acc_patches = metrics.balanced_accuracy_score(true_labels, pred_patches)
        outstr += f'\nPatches: {train_or_test} {epoch}, loss: {loss_patches:.6f}, {train_or_test} acc: {acc_patches:.6f},' \
                  f'  {train_or_test} avg acc: {avg_acc_patches:.6f}'
    if args.train_curves:
        loss_curves = loss_curves * 1.0 / count
        avg_acc_curves = metrics.balanced_accuracy_score(true_labels, pred_curves)
        outstr += f'\nCurves: {train_or_test} {epoch}, loss: {loss_curves:.6f}, {train_or_test} acc: {acc_curves:.6f},' \
                  f'  {train_or_test} avg acc: {avg_acc_curves:.6f}'
    io.cprint(outstr)


def apply_rsmix(args, data, device, label, label_b, lam, rsmix):
    if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (
            args.beta != 0.0):
        data = data.cpu().numpy()
    if args.rot:
        data = provider.rotate_point_cloud(data)
        data = provider.rotate_perturbation_point_cloud(data)
    if args.rdscale:
        tmp_data = provider.random_scale_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = tmp_data
    if args.shift:
        tmp_data = provider.shift_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = tmp_data
    if args.jitter:
        tmp_data = provider.jitter_point_cloud(data[:, :, 0:3])
        data[:, :, 0:3] = tmp_data
    if args.rddrop:
        data = provider.random_point_dropout(data)
    if args.shuffle:
        data = provider.shuffle_points(data)
    r = np.random.rand(1)
    if args.beta > 0 and r < args.rsmix_prob:
        rsmix = True
        data, lam, label, label_b = rsmix_provider.rsmix(data, label, beta=args.beta, n_sample=args.nsample,
                                                         KNN=args.knn)
    if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (
            args.beta != 0.0):
        data = torch.FloatTensor(data)
    if rsmix:
        lam = torch.FloatTensor(lam)
        lam, label_b = lam.to(device), label_b.to(device).squeeze()
    else:
        lam = None
        label_b = None
    return data, label, label_b, lam, rsmix


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) * 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def set_fixed_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    # load model patches
    model_patches = load_model(args, device)
    model_path_patches = os.path.join('pretrained',
                                      f'{args.model}_patches{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_patches == "" else args.model_path_patches
    checkpoint = torch.load(model_path_patches, map_location='cpu')
    model_patches.load_state_dict(checkpoint)

    # load model curves
    model_curves = load_model(args, device)
    model_path_curves = os.path.join('pretrained',
                                     f'{args.model}_curves{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_curves == "" else args.model_path_curves
    checkpoint = torch.load(model_path_curves, map_location='cpu')
    model_curves.load_state_dict(checkpoint)

    # load model random
    model_random = load_model(args, device)
    model_path_random = os.path.join('pretrained',
                                     f'{args.model}_random{"_wm.t7" if args.use_wolfmix else ".t7"}') if args.model_path_random == "" else args.model_path_random
    checkpoint = torch.load(model_path_random, map_location='cpu')
    model_random.load_state_dict(checkpoint)

    model_patches.eval()
    model_curves.eval()
    model_random.eval()

    def test_corrupt(args, split, model):
        with torch.no_grad():
            model_patches = model[0]
            model_curves = model[1]
            model_random = model[2]
            test_loader = DataLoader(ModelNetC(args=args, split=split),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            test_true = []
            test_pred = []
            set_fixed_seed(args)

            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()  # TODO: #B,
                data = data.permute(0, 2, 1)  # TODO:B,3,N
                anchors = farthest_point_sample(data.transpose(1, 2), args.k_tilde)  # TODO: (B,k_tilde)
                for cur_ppc in range(args.k_tilde):
                    random_ppc = extract_random(data, args.nr)  # B,3,nr Global partial point-cloud
                    patches_ppc = extract_patches(data, anchors[:, [cur_ppc]],
                                                  args.np)  # B,3,np Local partial point-cloud
                    curves_ppc = extract_curves(data, anchors[:, [cur_ppc]], args.m,
                                                args.nc)  # B,3,nc Local partial point-cloud
                    logits_random = model_random(random_ppc)
                    logits_patches = model_patches(patches_ppc)
                    logits_curves = model_curves(curves_ppc)
                    if cur_ppc == 0:
                        tot_pred_curves = logits_curves.unsqueeze(-2)
                        tot_pred_patches = logits_patches.unsqueeze(-2)
                        tot_pred_random = logits_random.unsqueeze(-2)
                    else:
                        tot_pred_curves = torch.cat((tot_pred_curves, logits_curves.unsqueeze(-2)), dim=-2)  # BxTxC
                        tot_pred_patches = torch.cat((tot_pred_patches, logits_patches.unsqueeze(-2)), dim=-2)  # BxTxC
                        tot_pred_random = torch.cat((tot_pred_random, logits_random.unsqueeze(-2)), dim=-2)  # BxTxC

                # TODO: change this if-else to torch.cat([large list of logits])
                tot_pred = torch.cat((tot_pred_curves, tot_pred_patches, tot_pred_random), dim=-2)
                logits = torch.mean(tot_pred, dim=-2)

                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}

    model = [model_patches, model_curves, model_random]
    eval_corrupt_wrapper(model, test_corrupt, {'args': args})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='EPiC: Ensemble of partial Point Clouds for robust classification - Official Implementation')
    # Training and testing parameters
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model if True, otherwise train')
    parser.add_argument('--exp_name', type=str, default='my_experiment', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['curvenet', 'dgcnn', 'dgcnn_v1', 'dgcnn_v2', 'dgcnn_v3', 'dgcnn_v5', 'gdanet', 'rpc',
                                 'pct', 'custom_model'],
                        help='Model Architecture to use as a basic model for random, patches and curves')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')

    # Architectures parameters
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    # EPiC parameters
    parser.add_argument('--nc', type=int, default=512, metavar='N',
                        help='number of points per curve')
    parser.add_argument('--np', type=int, default=512, metavar='N',
                        help='number of points per patch')
    parser.add_argument('--nr', type=int, default=128, metavar='N',
                        help='number of points per random')
    parser.add_argument('--m', type=int, default=20, metavar='N',
                        help='number of neighbors for curve random picking')
    parser.add_argument('--k_tilde', type=int, default=4,
                        help='number of anchor points in the shape')
    parser.add_argument('--train_random', action='store_true',
                        help='enable for random ppc training')
    parser.add_argument('--train_patches', action='store_true',
                        help='enable for patches ppc training')
    parser.add_argument('--train_curves', action='store_true',
                        help='enable for curves ppc training')

    parser.add_argument('--model_path_patches', type=str, default='', metavar='N',
                        help='Pretrained model path for patches testing')
    parser.add_argument('--model_path_random', type=str, default='', metavar='N',
                        help='Pretrained model path for random testing')
    parser.add_argument('--model_path_curves', type=str, default='', metavar='N',
                        help='Pretrained model path for curves testing')

    # Augmentations parameters
    parser.add_argument('--use_wolfmix', action='store_true', help='if to apply augmentation')
    # rsmix
    parser.add_argument('--rdscale', action='store_true', help='random scaling data augmentation')
    parser.add_argument('--shift', action='store_true', help='random shift data augmentation')
    parser.add_argument('--shuffle', action='store_true', help='random shuffle data augmentation')
    parser.add_argument('--rot', action='store_true', help='random rotation augmentation')
    parser.add_argument('--jitter', action='store_true', help='jitter augmentation')
    parser.add_argument('--rddrop', action='store_true', help='random point drop data augmentation')
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=1.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512,
                        help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--normal', action='store_true', help='use normal')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    # pointwolf
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
    parser.add_argument('--w_sample_type', type=str, default='fps',
                        help='Sampling method for anchor point, option : (fps, random)')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25,
                        help='Maximum translation range of local transformation')

    args = parser.parse_args()

    _init_()
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args)
