# -*- coding: utf-8 -*-
# @Time    : 2020/12/26 5:15 下午
# @Author  : Yijia Zheng
# @FileName: visualization.py

import joblib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 500

model_name = 'svm'

method = 'pca'

if model_name=='rf':
    model = joblib.load('save/my_model/rf_{}_val.pkl'.format(method))
    feature_importance = model.feature_importances_
    # feature_importance = sorted(feature_importance.keys())
    print(model.min_samples_split)
    key_list = []
    value_list = []
    for key, values in feature_importance.items():
        key_list.append(key)
        value_list.append(values)
    print(feature_importance)
    print(len(key_list))
    plt.hist(value_list, bins=20, color='#607c8e')
    plt.title('Feature Importance of {}'.format(method))
    plt.savefig('save/plots/feature_importance/{}.png'.format(method))
    plt.show()

else:
    model = joblib.load('save/my_model/svm_lasso_val.pkl')
    model.show()