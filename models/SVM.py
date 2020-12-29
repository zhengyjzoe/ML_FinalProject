# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 5:11 下午
# @Author  : Yijia Zheng
# @FileName: SVM.py

import logging
import random
import numpy as np
import copy
import time
import pickle


class SVM:
    def __init__(self, kernel='rbf', alg='SMO', gamma=None, C=1.0):
        self.eps = 1e-6
        self.alg = alg
        self.C, self.w, self.b, self.ksi = C, [], 0.0, []
        self.n_sv = -1
        self.sv_x, self.sv_y, self.alphas = np.zeros(0), np.zeros(0), np.zeros(0)
        self.kernel = kernel
        if self.kernel == 'rbf':
            self.gamma = gamma
        else:
            logging.basicConfig(level=logging.WARNING)

    def _kernel(self, x, z=None):
        if z is None:
            z = x
        if self.kernel == 'linear':
            return np.dot(x, z.T)
        elif self.kernel == 'rbf':
            xx, zz = np.sum(x*x, axis=1), np.sum(z*z, axis=1)
            res = -2.0 * np.dot(x, z.T) + xx.reshape((-1, 1)) + zz.reshape((1, -1))
            del xx, zz
            return np.exp(-self.gamma * res)
        else:
            print('Unknown kernel')
            exit(3)


    def _SMO(self, K, y):

        def choose_alphas():  # choose alpha heuristically
            check_all_examples = False  # whether or not need to check all examples
            # passes, max_passes = 0, 2
            # while passes < max_passes:
            while True:
                num_changed_alphas = 0
                if check_all_examples:
                    # in range_i, unbounded alphas rank first
                    range_i = range(m)
                else:
                    # check unbounded examples only
                    range_i = [i for i in range(m) if self.eps < self.alphas[i] < self.C - self.eps]
                for i in range_i:
                    yi, ai_old = y[i], self.alphas[i]
                    Ei = np.sum(self.alphas * y * K[i]) + self.b - yi
                    logging.debug('Ei = ' + str(Ei))
                    if (yi*Ei < -tol and ai_old < self.C) or (yi*Ei > tol and ai_old > 0):
                        range_j = list(range(m))
                        random.shuffle(range_j)
                        for j in range_j:
                            if j == i:
                                continue
                            yj = y[j]
                            Ej = np.sum(self.alphas * y * K[j]) + self.b - yj
                            yield (i, j, Ei, Ej)
                            if updated[0]:  # if (i, j) pair changed in the latest iteration
                                num_changed_alphas += 1
                                break
                if num_changed_alphas == 0:
                    if check_all_examples:  # if have checked all examples and no alpha violates KKT condition
                        break
                    else:
                        check_all_examples = True  # check all examples in the __next__ iteration as a safeguard
                else:
                    check_all_examples = False
            yield -1, -1, 0.0, 0.0

        print('Begin SMO...')
        m = len(y)
        self.alphas, self.b = np.zeros(m), 0.0
        tol = self.eps
        logging.debug('m = ' + str(m))
        gen = choose_alphas()
        n_iter = 0
        updated = [False]  # use mutable object to pass message between functions
        while True:
            n_iter += 1
            logging.info('Iteration ' + str(n_iter))
            # run over pair (i, j).  But for some alpha_i, only choose one alpha_j in an iteration epoch.
            try:
                i, j, Ei, Ej = gen.__next__()
            except StopIteration:
                break
            if i == -1:  # no more (i, j) pairs against KKT condition
                break
            updated[0] = False
            yi, yj, ai_old, aj_old = y[i], y[j], self.alphas[i], self.alphas[j]
            if yi != yj:
                L, H = max(0.0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
            else:
                L, H = max(0.0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
            logging.debug('L = ' + str(L) + ', H = ' + str(H))
            if H - L < self.eps:
                continue
            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            logging.debug('eta = ' + str(eta))
            if eta <= 0:  # This should not be happen, because gram matrix should be PSD
                if eta == 0.0:
                    print('eta = 0, possibly identical examples encountered!')
                else:
                    print('GRAM MATRIX IS NOT PSD!')
                    exit(0)
                continue
            aj_new = aj_old + yj * (Ei - Ej) / eta
            if aj_new > H:
                aj_new = H
            elif aj_new < L:
                aj_new = L
            delta_j = aj_new - aj_old
            if abs(delta_j) < 1e-5:
                continue
            ai_new = ai_old + yi * yj * (aj_old - aj_new)
            delta_i = ai_new - ai_old
            self.alphas[i], self.alphas[j] = ai_new, aj_new
            bi = self.b - Ei - yi * delta_i * K[i, i] - yj * delta_j * K[i, j]
            bj = self.b - Ej - yi * delta_i * K[i, j] - yj * delta_j * K[j, j]
            if 0 < ai_new < self.C:
                self.b = bi
            elif 0 < aj_new < self.C:
                self.b = bj
            else:
                self.b = (bi + bj) / 2.0
            updated[0] = True
            # logging.info('Updated through' + str(i) + str(j))
            # logging.debug('alphas:' + str(self.alphas))

        print('Finish SMO...')
        return self.alphas, self.b

    def fit(self, x, y):
        assert type(x) == np.ndarray
        print(x.shape, y.shape)
        # In the design matrix x: m examples, n features
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = 1.0 / x.shape[1]
            print('gamma = ', self.gamma)
        assert self.alg == 'SMO'
        K = self._kernel(x)
        self._SMO(K, y)

        logging.info('self.alphas = ' + str(self.alphas))
        sv_indices = list(filter(lambda i:self.alphas[i] > self.eps, range(len(y))))
        self.sv_x, self.sv_y, self.alphas = x[sv_indices], y[sv_indices], self.alphas[sv_indices]
        self.n_sv = len(sv_indices)
        logging.info('sv_indices:' + str(sv_indices))
        print(self.n_sv, 'SVs!')
        logging.info(str(np.c_[self.sv_x, self.sv_y]))
        if self.kernel == 'linear':
            self.w = np.dot(self.alphas * self.sv_y, self.sv_x)
        if self.alg == 'dual':
            # calculate b: average over all support vectors
            sv_boundary = self.alphas < self.C - self.eps
            self.b = np.mean(self.sv_y[sv_boundary] - np.dot(self.alphas * self.sv_y,
                                                             self._kernel(self.sv_x, self.sv_x[sv_boundary])))

    def predict_score(self, x):
        return np.dot(self.alphas * self.sv_y, self._kernel(self.sv_x, x)) + self.b

    def show(self):
        if (self.alg == 'dual') or (self.alg == 'SMO'):
            print('\nFitted parameters:')
            print('n_sv = ', self.n_sv)
            print('sv_x = ', self.sv_x)
            print('sv_y = ', self.sv_y)
            print('alphas = ', self.alphas)
            if self.kernel == 'linear':
                print('w = ', self.w)
            print('b = ', self.b)
        else:
            print('No known optimization method!')

    def predict(self, x):
        return np.sign(self.predict_score(x))

    def save(self, file_name='BinarySVM1.pkl'):
        fh = open('../model/' + file_name, 'wb')
        pickle.dump(self, fh)
        fh.close()


class MultiSVM:
    def __init__(self, kernel='rbf', alg='SMO', decision_function='ovo', gamma=None, degree=None, C=1.0):
        self.degree, self.gamma, self.decision_function = degree, gamma, decision_function
        self.alg, self.C = alg, C
        self.kernel = kernel
        self.n_class, self.classifiers = 0, []

    def fit(self, x, y):
        labels = np.unique(y)
        self.n_class = len(labels)
        print(labels)
        if self.decision_function == 'ovr':  # one-vs-rest method
            for label in labels:
                y1 = np.array(y)
                y1[y1 != label] = -1.0
                y1[y1 == label] = 1.0
                print ('Begin training for label', label, 'at',
                    time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time())))
                t1 = time.time()
                clf = SVM(self.kernel, self.alg, self.gamma, self.degree, self.C)
                clf.fit(x, y1)
                t2 = time.time()
                print ('Training time for ' + str(label) + '-vs-rest:', t2 - t1, 'seconds')
                self.classifiers.append(copy.deepcopy(clf))
        else:  # use one-vs-one method
            assert self.decision_function == 'ovo'
            n_labels = len(labels)
            for i in range(n_labels):
                for j in range(i+1, n_labels):
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    x1, y1 = np.r_[x[neg_id], x[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.0
                    y1[y1 == labels[j]] = 1.0
                    print ('Begin training classifier for label', labels[i], 'and label', labels[j], 'at',
                        time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time())))
                    t1 = time.time()
                    clf = SVM(self.kernel, self.alg, self.gamma, self.C)
                    clf.fit(x1, y1)
                    t2 = time.time()
                    print ('Training time for ' + str(labels[i]) + '-vs-' + str(labels[j]) + ':', t2 - t1, 'seconds')
                    self.classifiers.append(copy.deepcopy(clf))

    def predict(self, test_data):
        n_samples = test_data.shape[0]
        if self.decision_function == 'ovr':
            score = np.zeros((n_samples, self.n_class))
            for i in range(self.n_class):
                clf = self.classifiers[i]
                score[:, i] = clf.predict_score(test_data)
            return np.argmax(score, axis=1)
        else:
            assert self.decision_function == 'ovo'
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i+1, self.n_class):
                    res = self.classifiers[clf_id].predict(test_data)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    clf_id += 1
            return np.argmax(vote, axis=1)

    def save(self, file_name='svm_pca_val.pkl'):
        fh = open('../save/my_model/' + file_name, 'wb')
        pickle.dump(self, fh)
        fh.close()

    def show(self):
        for clf in self.classifiers:
            clf.show()

if __name__ == '__main__':
    pass