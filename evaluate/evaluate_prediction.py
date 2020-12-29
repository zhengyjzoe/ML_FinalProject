# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 8:13 下午
# @Author  : Yijia Zheng
# @FileName: evaluate_prediction.py

import sys
sys.path.append('..')
import numpy as np
from preprocessing.data import load_data, load_reduction_data
from models.tools import plot_confusion_matrix, confusion_matrix
from sklearn import metrics

DIM = 40
MODEL_NAME = 'mlp' # choosing from knn, svm, rf, mlp
Reduction_Method = 'pca' # choosing from pca, lasso, original

#load data
X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')

result_path = '../save/result/{}_{}_val.npy'.format(MODEL_NAME, Reduction_Method)
plots_path = '../save/plots/confusion_matrix/{}_{}.png'.format(MODEL_NAME, Reduction_Method)

y_pred = np.load(result_path)
if MODEL_NAME=='mlp':
    y_pred = np.argwhere(y_pred==1)[:,1]
conf_matrix = metrics.confusion_matrix(y_val, y_pred)
plot_confusion_matrix(conf_matrix, plots_path)
print('Acc:', metrics.accuracy_score(y_pred, y_val))
print('Micro:', metrics.precision_score(y_val, y_pred, average='micro'))
print('Micro:', metrics.precision_score(y_val, y_pred, average='macro'))






