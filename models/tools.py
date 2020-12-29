# -*- coding: utf-8 -*-
# @Time    : 2020/11/28 1:58 下午
# @Author  : Yijia Zheng
# @FileName: tools.py

# Useful Functions that will be repeatedly used
import numpy as np
import matplotlib.pyplot as plt
import itertools


def pca(XMat, k):
    average = np.mean(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec=  np.linalg.eig(covX)
    index = np.argsort(-featValue)
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]]) #转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData.real, reconData


def confusion_matrix(preds, labels):
    conf_matrix = np.zeros((len(labels), len(labels)))
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(confusion_mat, save_path):
    plt.rcParams['figure.dpi'] = 500
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
    thresh = confusion_mat.max() / 2.
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, confusion_mat[i, j],
                 horizontalalignment="center",
                 color="black" if confusion_mat[i, j] > thresh else "white")
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(confusion_mat.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()



