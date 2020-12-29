# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 12:32 下午
# @Author  : Yijia Zheng
# @FileName: temp.py

###一些临时的查看代码或测试代码

# import joblib
# view_root = 'save/rf_test.pkl'
# model = joblib.load(view_root)
# print(dir(model))
# print(model.n_estimators)

import sys
sys.path.append('..')

from preprocessing.data import load_data, load_reduction_data
X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')



# X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')
# # print(len(y_val))
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# err = np.load('train_Lasso/save/FISTA_smooth_w.npy')
# print(len(np.where(err<1e-4)[0]))
# plt.plot(err)
# plt.show()

# 查看PCA的维度取值
# from preprocessing.data import load_data
# X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')
# from sklearn.decomposition import PCA
#
# pca = PCA()
# pca.fit(X_train)
# var_ratio = pca.explained_variance_ratio_
# ratio = [sum(var_ratio[0:i]) for i in range(len(var_ratio))]
# plt.plot(ratio[0:100])
# plt.ylabel('Accumulated Variance Ratio')
# plt.xlabel('Dim')
# plt.title('Accumulated Variance Ratio of PCA')
# plt.show()




