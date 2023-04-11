#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from augmentation.PointWOLF.PointWOLF import PointWOLF


def download():
    print("download!!! data")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(
            os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    N, C = pointcloud.shape
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[C])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[C])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):
    def __init__(self, args, partition='train'):
        self.data, self.label = load_data(partition)
        self.args = args
        self.partition = partition
        if self.args.use_wolfmix:
            self.PointWOLF = PointWOLF(args)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.args.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
            if self.args.use_wolfmix:
                _, pointcloud = self.PointWOLF(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
