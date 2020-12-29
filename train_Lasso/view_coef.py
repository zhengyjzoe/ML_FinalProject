# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 5:46 下午
# @Author  : Yijia Zheng
# @FileName: view_coef.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOSS_NAME = 'smooth'
OBJECT = 'w'


plt.rcParams['figure.figsize'] = (8.0, 5.0)
plt.rcParams['image.interpolation'] = 'nearest'

if LOSS_NAME == 'smooth':
    for idx, (alg_name, color) in enumerate(zip(['BASIC', 'FISTA', 'Nesterov'], ['#87CEFA', '#FFA07A', '#778899'])):
        path = 'save/{}_{}_{}.npy'.format(alg_name, LOSS_NAME, OBJECT)
        data = np.load(path)
        print(len(np.where(np.abs(data)>1e-1)[0]))
        # plt.hist(data, bins=100, alpha=0.5, color=color, label=alg_name)
        sns.kdeplot(data, shade=True, color=color, label=alg_name)

else:
    for idx, (alg_name, color) in enumerate(zip(['BASIC', 'FISTA', 'Nesterov'], ['#87CEFA', '#FFA07A', '#778899'])):
        path = 'save/{}_{}.npy'.format(alg_name, OBJECT)
        data = np.load(path)
        print(len(np.where(np.abs(data) < 1e-1)[0]))
        # plt.hist(data, bins=100, alpha=0.5, color=color, label=alg_name)
        sns.kdeplot(data, shade=True, color=color, label=alg_name)

plt.legend(loc='best')
plt.title('Coefficient distribution of {}'.format(LOSS_NAME))
plt.savefig('Plots/{}_distribution.jpg'.format(LOSS_NAME), dpi=200)
plt.show()


