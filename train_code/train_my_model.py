# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 12:44 下午
# @Author  : Yijia Zheng
# @FileName: train_my_model.py

import sys
sys.path.append('..')
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

#### import models and tools####
from preprocessing.data import load_data, load_reduction_data
from models.tools import pca, plot_confusion_matrix
from models.SVM import MultiSVM
from models.KNN import KNN
from models.BPNN import NeuralNet
from models.RF import RandomForestClassifier




DIM = 60
MODEL_NAME = 'mlp' # choosing from knn, svm, rf, mlp
Reduction_Method = 'lasso' # choosing from pca, lasso, original

result_save_path = '../save/result/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

#load processed_data
if Reduction_Method=='pca':
    X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')
    X_train = np.asarray(pca(X_train, DIM)[0])
    X_val = np.asarray(pca(X_val, DIM)[0])
    X_test = np.asarray(pca(X_test, DIM)[0])

elif Reduction_Method=='lasso':
    X_train, X_val, X_test, y_train, y_val, y_test = load_reduction_data('r50')

else:
    assert Reduction_Method=='original'
    X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')

print('Data process done!')



#######  Train Model ######

if MODEL_NAME == 'svm':
    clf = MultiSVM(C=1, kernel='rbf', gamma=1/X_train.shape[1], decision_function='ovo')
    clf.fit(X_train, y_train)
    clf.save('svm_{}_val.pkl'.format(Reduction_Method))
    y_pred = clf.predict(X_val)
    # np.save(result_save_path+'{}_{}_val'.format(MODEL_NAME, Reduction_Method), y_pred)


elif MODEL_NAME == 'knn':
    model = KNN(X_train, y_train)
    neighbor_list = np.linspace(10, 10, 1)
    best_acc = 0
    # for neighbor_num in map(int, list(neighbor_list)):
    #     y_val_predicted = []
    #     for idx, sample in enumerate(X_val):
    #         print(idx)
    #         y_val_predicted.append(model.fit(sample, neighbor_num))
    #     result = np.vstack((np.asarray(y_val_predicted), y_val)).T
    #     accuracy = len(result[np.where(result[:, 0]==result[:, 1])])/result.shape[0]
    #     print('acc=', accuracy)
    #     if accuracy > best_acc:
    #         best_model = model
    #         best_result = result
    #         best_acc = accuracy
    #         best_num_neighbor = neighbor_num
    #     np.save(result_save_path + '{}_{}_val'.format(MODEL_NAME, Reduction_Method), y_val_predicted)
    for neighbor_num in map(int, list(neighbor_list)):
        y_train_predicted = []
        for idx, sample in enumerate(X_train):
            print(idx)
            y_train_predicted.append(model.fit(sample, neighbor_num))
        result = np.vstack((np.asarray(y_train_predicted), y_train)).T
        accuracy = len(result[np.where(result[:, 0]==result[:, 1])])/result.shape[0]
        print('acc=', accuracy)
        if accuracy > best_acc:
            best_model = model
            best_result = result
            best_acc = accuracy
            best_num_neighbor = neighbor_num

    print('Best {} is from {} neighbors'.format(best_acc, best_num_neighbor))


elif MODEL_NAME == 'rf':
    clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                 min_samples_split=2,
                                 min_samples_leaf=3,
                                 min_split_gain=0.0,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=2018111396)
    clf.fit(pd.DataFrame(X_train), pd.DataFrame(y_train).iloc[:, 0])
    joblib.dump(clf, '../save/my_model/rf_{}_val.pkl'.format(Reduction_Method))
    y_pred = clf.predict(pd.DataFrame(X_val))
    # np.save(result_save_path + '{}_{}_val'.format(MODEL_NAME, Reduction_Method), y_pred)

elif MODEL_NAME == 'mlp':
    out_num = np.max(y_train) + 1
    train_num = len(y_train)
    val_num = len(y_val)
    y_train_sparse = np.zeros((train_num, out_num))
    y_train_sparse[np.arange(0, train_num), y_train] = 1
    y_val_sparse = np.zeros((val_num, out_num))
    y_val_sparse[np.arange(0, val_num), y_val] = 1

    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    plt.rcParams['image.interpolation'] = 'nearest'

    for (act_name, alg_name, color) in zip(['relu', 'relu', 'sigmoid', 'sigmoid'],
                                    ['sgd', 'adam', 'sgd', 'adam'], ['#ADD8E6', '#F08080', '#FAFAD2', '#90EE90']):
        model = NeuralNet(hidden_layers=(100,), batch_size=200, learning_rate=1e-3,
                          max_iter=100, tol=1e-4, alpha=1e-5,
                          activation=act_name, solver=alg_name)
        model.fit(X_train, y_train_sparse)
        y_pred = model.predict(X_val)
        # np.save(result_save_path + '{}_{}_{}_{}_val'.format(MODEL_NAME, Reduction_Method, act_name, alg_name), y_pred)
        print('accuracy:', 1 - np.sum(np.abs(y_pred - y_val_sparse)) / 2 / y_val_sparse.shape[0])
        ax1 = plt.subplot(211)
        plt.plot(model.loss_curve, color=color, label=alg_name+'_'+act_name)
        plt.legend(loc='best')
        plt.title('Loss Curve in Training Set')
        ax2 = plt.subplot(212)
        plt.plot(model.acc_curve, color=color, label=alg_name + '_' + act_name)
    plt.legend(loc='best')
    plt.title('Accuracy Curve in Training Set')
    # plt.savefig('../save/plots/mlp_training.jpg', dpi=200)
    plt.show()


else:
    print('Misuse, please choose from svm, knn, rf, mlp')