# -*- coding: utf-8 -*-
# @Time    : 2020/11/28 1:26 下午
# @Author  : Yijia Zheng
# @FileName: train_sklearn_model.py

## used for comparition

import sys
sys.path.append('..')
from preprocessing.data import load_data, load_reduction_data
from models.tools import pca

# import SVM as svm
from sklearn import svm
import numpy as np
from sklearn import metrics
import joblib

DIM = 40
MODEL_NAME = 'svm' # choosing from knn, svm, rf, mlp
Reduction_Method = 'pca' # choosing from pca, lasso, original

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
    clf = svm.SVC(C=1, kernel='rbf', gamma=1/X_train.shape[1], decision_function_shape='ovo')
    clf.fit(X_train, y_train.ravel())
    print(clf.score(X_train, y_train))  # 精度
    joblib.dump(clf, '../save/test_model/svm_val.pkl')
    y_pred = clf.predict(X_val)
    y_train_pred = clf.predict(X_train)
    cm_train = metrics.confusion_matrix(y_train, y_train_pred)  # 训练集混淆矩阵
    cm_val = metrics.confusion_matrix(y_val, y_pred)  # 测试集混淆矩阵
    print(cm_train)
    print(cm_val)
    print(metrics.accuracy_score(y_train, y_train_pred))
    print(metrics.accuracy_score(y_val, y_pred))


elif MODEL_NAME == 'knn':
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)


elif MODEL_NAME == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(bootstrap=True, oob_score=True, criterion='gini', n_estimators=100, min_samples_split=2,
                                 min_samples_leaf=3, random_state=2018111396)
    clf.fit(X_train, y_train)
    joblib.dump(clf, '../save/test_model/rf_val.pkl')
    y_pred = clf.predict(X_val)
    y_train_pred = clf.predict(X_train)
    cm_train = metrics.confusion_matrix(y_train, y_train_pred)  # 训练集混淆矩阵
    cm_val = metrics.confusion_matrix(y_val, y_pred)  # 测试集混淆矩阵
    print(cm_train)
    print(cm_val)
    print(metrics.accuracy_score(y_train, y_train_pred))
    print(metrics.accuracy_score(y_val, y_pred))


elif MODEL_NAME == 'mlp':
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(100), activation='relu',
                        batch_size=200, learning_rate_init=1e-3, solver='adam', learning_rate='constant')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_train_pred = clf.predict(X_train)
    cm_train = metrics.confusion_matrix(y_train, y_train_pred)  # 训练集混淆矩阵
    cm_val = metrics.confusion_matrix(y_val, y_pred)  # 测试集混淆矩阵
    print(cm_train)
    print(cm_val)
    print(metrics.accuracy_score(y_train, y_train_pred))
    print(metrics.accuracy_score(y_val, y_pred))


else:
    print('Misuse, please choose from svm, knn, rf, mlp ')

print('Acc:', metrics.accuracy_score(y_pred, y_val))
print('Micro:', metrics.precision_score(y_val, y_pred, average='micro'))
print('Micro:', metrics.precision_score(y_val, y_pred, average='macro'))