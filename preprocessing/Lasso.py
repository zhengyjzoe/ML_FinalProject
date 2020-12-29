# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 11:45 上午
# @Author  : Yijia Zheng
# @FileName: Lasso.py


import numpy as np
from data import load_data
import os

CORF_ROOT = '../train_Lasso/save/'
alg_name = 'FISTA'

coef = np.load(CORF_ROOT + alg_name + '_w.npy') # coef.shape = (2048,)
print('Length of new coef =' ,len(np.where(abs(coef)>1e-1)[0]))
X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')
X_train = X_train[:, np.where(abs(coef) > 1e-1)[0]]
X_val = X_val[:, np.where(abs(coef) > 1e-1)[0]]
X_test = X_test[:, np.where(abs(coef) > 1e-1)[0]]

if not os.path.exists('../data/data_reduction/'):
    os.makedirs('../data/data_reduction/')

np.save('../data/data_reduction/{}_train'.format(alg_name), X_train)
np.save('../data/data_reduction/{}_val'.format(alg_name), X_val)
np.save('../data/data_reduction/{}_test'.format(alg_name), X_test)

