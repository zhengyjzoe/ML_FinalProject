# -*- coding: utf-8 -*-
# @Time    : 2020/11/28 2:03 下午
# @Author  : Yijia Zheng
# @FileName: data.py


import sys
sys.path.append("..")
import torch
import numpy as np
import random

random.seed(2018111396)

embeddings_dir = '../data/feature_map/'
data_dir = '../data/data_reduction/'
labels_dir = embeddings_dir


def load_data(feature_extractor):

    embeddings = load_embeddings(feature_extractor)
    labels = load_labels(feature_extractor)
    X_train, X_temp, y_train, y_temp = generate_splits(embeddings, labels, test_size=0.2)
    X_val, X_test, y_val, y_test = generate_splits(X_temp, y_temp, test_size=0.5)
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_reduction_data(feature_extractor, optim_alg='FISTA'):

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(feature_extractor)
    X_train = np.load(data_dir + '{}_train.npy'.format(optim_alg))
    X_val = np.load(data_dir + '{}_val.npy'.format(optim_alg))
    X_test = np.load(data_dir + '{}_test.npy'.format(optim_alg))

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_embeddings(feature_extractor):

    filename = 'embeddings_' + feature_extractor + '.pt'
    path = embeddings_dir + filename
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = torch.load(path, map_location=torch.device(device))
    X = embeddings.cpu().numpy()
    return X


def load_labels(feature_extractor):

    filename = 'labels_' + feature_extractor + '.pt'
    path = labels_dir + filename
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = torch.load(path, map_location=torch.device(device))
    y = labels.cpu().numpy()
    return y


def split_data(X, y, test_size):

    train_idx = np.array(random.sample(range(X.shape[0]), int(X.shape[0] * (1-test_size))))
    test_idx = np.delete(np.arange(0, len(y)), train_idx)
    train_x, train_y = np.asarray(X[train_idx]), np.asarray(y[train_idx])
    test_x, test_y = np.asarray(X[test_idx]), np.asarray(y[test_idx])

    return train_x, test_x, train_y, test_y


def generate_splits(X, y, test_size):

    return split_data(X, y, test_size=test_size)


def normalize_data(X_train, X_val, X_test):

    def _norm(x, mean, std):
        return (x-mean)/std

    x_mean = np.mean(X_train)
    x_std = np.std(X_train)
    X_train_transformed = _norm(X_train, x_mean, x_std)
    X_val_transformed = _norm(X_val, x_mean, x_std)
    X_test_transformed = _norm(X_test, x_mean, x_std)
    return X_train_transformed, X_val_transformed, X_test_transformed




if __name__=='__main__':
    import collections
    import matplotlib.pyplot as plt
    X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')
    print(X_train.shape)
    print(X_val.shape)
    print(collections.Counter(y_train), collections.Counter(y_val), collections.Counter(y_test))
    # labels = load_labels('r50')
    # print(collections.Counter(labels))
    # print(len(labels))
    # plt.show()
