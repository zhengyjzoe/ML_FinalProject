# -*- coding: utf-8 -*-
# @Time    : 2020/11/28 1:12 下午
# @Author  : Yijia Zheng
# @FileName: KNN.py

import math
import numpy as np
import collections
import operator


class KNN():

    def __init__(self, training_set, training_label, downsample = False):

        self.neighbors = []
        self.classVoters = {}
        self.training_set = training_set
        self.training_label = training_label
        self.downsample = downsample
        self.distance = []

    def _euclideanDistance(self, instance1, instance2):
        _distance = np.sum((instance1 - instance2) ** 2)
        return math.sqrt(_distance)

    def fit(self, test_instance, k):
        self.k = k
        # for x in range(self.training_set.shape[0]):
        #     dist = self._euclideanDistance(test_instance, self.training_set[x])
        #     self.distance.append((self.training_label[x], self.training_set[x], dist))
        dist = np.sum((test_instance-self.training_set)**2, axis=1)**0.5
        self.k_labels = [self.training_label[index] for index in dist.argsort()[0: k]]
        # self.distance = ()
        # print(self.distance)
        # self.distance.sort(key=operator.itemgetter(2))
        # distance = np.asarray(list(map(lambda x: x[2], self.distance)))
        # labels = np.asarray(list(map(lambda x: x[0], self.distance)))
        # features = np.asarray(list(map(lambda x: x[1], self.distance)))
        # return distance[0:k], labels[0:k], features[0:k]
        if self.downsample == True:
            label = self.k_labels[1]
        else:
            label = collections.Counter(self.k_labels).most_common(1)[0][0]
        return label



if __name__=='__main__':
    feature = np.array([[1,2,3],[2,3,4],[10,11,12]])
    label = np.array([1,3,3])
    test = np.array([5,6,7])
    knn = KNN(feature, label, True)
    print(knn.fit(test, 2))
    # print(knn.getResponse())