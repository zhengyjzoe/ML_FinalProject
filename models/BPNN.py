# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 2:09 下午
# @Author  : Yijia Zheng
# @FileName: BPNN.py

import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings

np.random.seed(2018111396)

def softmax(x):
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x


def sigmoid(x):
    exp_x = np.exp(-x)
    return 1 / (1 + exp_x)


def relu(x):
    relu_x = (np.abs(x) + x) / 2
    return relu_x


class NeuralNet(object):

    def __init__(self, hidden_layers, batch_size,
                 activation, learning_rate, max_iter,
                 solver, tol, alpha):

        hidden_layers = list(hidden_layers)
        self.hidden_layers = hidden_layers
        self.n_layers = len(hidden_layers)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = 1e-8
        self.solver = solver
        self.alpha = alpha

        # choose activation function for hidden layers
        if activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'relu':
            self.activation = relu

    def fit(self, X, y):
        self._fit(X, y)

    def predict(self, X):
        activations = []
        for i in range(self.n_layers):
            activations.append(np.empty((X.shape[0], self.hidden_layers[i]), np.float))
        activations.append(np.empty((self.batch_size, self.noutputs,), np.float))
        activations = self.__forward(X, activations)
        a = activations[self.n_layers]
        y = a / np.max(a, axis=1, keepdims=True)
        y[y < 1] = 0
        return y

    def _fit(self, X, y):
        X, y = self.__Check_X_y(X, y)
        nsamples, nfeatures = X.shape
        self.noutputs = y.shape[1]
        print("train samples num:%d," % (nsamples), "features num:%d," % (nfeatures), "output num:%d" % (self.noutputs))

        # 初始化参数：weights, bias, grad_weights, grad_bias
        # 备注：参数的初始化很重要，第一种初始化方式，其训练精度最高只有84%左右，而且训练周期比较长；
        #     第二种初始化方法，训练精度高，可达到97%以上，训练周期短，效率高。
        self.W = []
        self.b = []
        in_num = nfeatures
        out_num = self.hidden_layers[0]
        init_bound = np.sqrt(2.0 / (in_num + out_num))
        self.W.append(np.random.uniform(-init_bound, init_bound, (in_num, out_num)))
        self.b.append(np.random.uniform(-init_bound, init_bound, out_num))
        for i in range(self.n_layers - 1):
            in_num = self.hidden_layers[i]
            out_num = self.hidden_layers[i + 1]
            init_bound = np.sqrt(2.0 / (in_num + out_num))
            self.W.append(np.random.uniform(-init_bound, init_bound, (in_num, out_num)))
            self.b.append(np.random.uniform(-init_bound, init_bound, out_num))
        in_num = self.hidden_layers[self.n_layers - 1]
        out_num = self.noutputs
        init_bound = np.sqrt(2.0 / (in_num + out_num))
        self.W.append(np.random.uniform(-init_bound, init_bound, (in_num, out_num)))
        self.b.append(np.random.uniform(-init_bound, init_bound, out_num))

        # to save activation functions, will be used when calculating grads
        activations = []
        for i in range(self.n_layers):
            activations.append(np.empty((self.batch_size, self.hidden_layers[i]), np.float))
        activations.append(np.empty((self.batch_size, self.noutputs,), np.float))

        # to save grads of parameters
        self.grad_W = copy.deepcopy(self.W)
        self.grad_b = copy.deepcopy(self.b)

        # to save params with min-loss when training
        self.best_W = copy.deepcopy(self.W)
        self.best_b = copy.deepcopy(self.b)
        self.best_loss = np.inf
        self.loss_curve = []
        self.acc_curve = []

        # choose algorithms for grad optimization
        if self.solver == 'adam':
            # adam参数初始化(Adaptive Moment Estimation)
            self.iter_count = 0
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.mt_W = copy.deepcopy(self.W)
            self.mt_b = copy.deepcopy(self.b)
            self.vt_W = copy.deepcopy(self.W)
            self.vt_b = copy.deepcopy(self.b)
            for i in range(self.n_layers + 1):
                self.mt_W[i][:] = 0.0
                self.mt_b[i][:] = 0.0
                self.vt_W[i][:] = 0.0
                self.vt_b[i][:] = 0.0
        elif self.solver == 'sgd':
            # NAG参数初始化(nesterov accelerated gradient)
            self.momentum = 0.9
            self.velocities_W = copy.deepcopy(self.W)
            self.velocities_b = copy.deepcopy(self.b)
            for i in range(self.n_layers + 1):
                self.velocities_W[i][:] = 0.0
                self.velocities_b[i][:] = 0.0

        print('start training...')
        try:
            # epoch loop
            for epoch_count in range(self.max_iter):
                # 打乱数据
                shuffle_index = np.random.permutation(nsamples)
                shuffle_X = X[shuffle_index]
                shuffle_y = y[shuffle_index]
                # batch loop
                accumulated_loss = 0.0
                for i in range(0, nsamples - self.batch_size + 1, self.batch_size):
                    # 抽取批量数据
                    batch_X = shuffle_X[np.arange(i, i + self.batch_size)]
                    batch_y = shuffle_y[np.arange(i, i + self.batch_size)]

                    if self.solver == 'sgd':
                        for j in range(self.n_layers + 1):
                            self.W[j] += self.momentum * self.velocities_W[j]
                            self.b[j] += self.momentum * self.velocities_b[j]

                        activations = self.__forward(batch_X, activations)
                        self.__backpro(batch_X, batch_y, activations)
                        self.__update_params()

                        activations = self.__forward(batch_X, activations)
                        batch_loss = self.__compute_loss(activations[self.n_layers], batch_y)
                        accumulated_loss += (batch_loss / self.batch_size)

                    elif self.solver == 'adam':
                        activations = self.__forward(batch_X, activations)
                        self.__backpro(batch_X, batch_y, activations)
                        self.__update_params()

                        activations = self.__forward(batch_X, activations)
                        batch_loss = self.__compute_loss(activations[self.n_layers], batch_y)
                        accumulated_loss += (batch_loss / self.batch_size)

                # 计算本轮epoch损失，训练精度
                loss = accumulated_loss
                pre_y = self.predict(X)
                accuracy = 1 - np.sum(np.abs(y - pre_y)) / 2.0 / nsamples
                print('%dth-epoch-loss:' % (epoch_count), loss, 'accuracy:', accuracy)

                self.loss_curve.append(loss)
                self.acc_curve.append(accuracy)
                self.__update_no_improvement_count()
                if self.no_improvement_count > 2:
                    print('Training loss did not improve more than tol=%f'
                          'for two consecutive epochs.' % self.tol)
                    self.W = self.best_W
                    self.b = self.best_b
                    break
                if epoch_count + 1 == self.max_iter:
                    warnings.warn('Stochastic Optimizer: Maximum iterations (%d) '
                                  'reached and the optimization hasn\'t converged yet.'
                                  % self.max_iter)
        except KeyboardInterrupt:
            warnings.warn('Training interrupted by user.')

    def __update_no_improvement_count(self):
        #检查是否达到停止训练的条件
        if self.loss_curve[-1] > self.best_loss - self.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        if self.loss_curve[-1] < self.best_loss:
            self.best_loss = self.loss_curve[-1]
            self.best_W = self.W.copy()
            self.best_b = self.b.copy()


    def __forward(self, X, activations):

        activations[0] = self.activation(np.dot(X, self.W[0]) + self.b[0])
        for i in range(self.n_layers - 1):
            activations[i + 1] = self.activation(np.dot(activations[i], self.W[i + 1]) + self.b[i + 1])
        activations[self.n_layers] = softmax(np.dot(activations[self.n_layers - 1],
                                                    self.W[self.n_layers]) + self.b[self.n_layers])
        return activations

    def __backpro(self, X, y, activations):

        # 计算梯度
        if self.activation == relu:
            # relu
            a = activations[self.n_layers]
            de_da = -y / (a + self.epsilon) / self.batch_size
            de_dz = a * (de_da - np.sum(a * de_da, axis=1, keepdims=True))
            for i in range(self.n_layers):
                self.grad_W[self.n_layers - i] = np.dot(activations[self.n_layers - i - 1].T, de_dz)
                self.grad_b[self.n_layers - i] = np.sum(de_dz, axis=0)
                de_da = np.dot(de_dz, self.W[self.n_layers - i].T)
                da_dz = copy.deepcopy(activations[self.n_layers - i - 1])
                da_dz[da_dz > 0] = 1
                de_dz = de_da * da_dz
            self.grad_W[0] = np.dot(X.T, de_dz)
            self.grad_b[0] = np.sum(de_dz, axis=0)
        elif self.activation == sigmoid:
            # sigmoid
            a = activations[self.n_layers]
            de_da = -y / (a + self.epsilon) / self.batch_size
            de_dz = a * (de_da - np.sum(a * de_da, axis=1, keepdims=True))
            for i in range(self.n_layers):
                self.grad_W[self.n_layers - i] = np.dot(activations[self.n_layers - i - 1].T, de_dz)
                self.grad_b[self.n_layers - i] = np.sum(de_dz, axis=0)
                de_da = np.dot(de_dz, self.W[self.n_layers - i].T)
                a = activations[self.n_layers - i - 1]
                da_dz = a * (1 - a)
                de_dz = de_da * da_dz
            self.grad_W[0] = np.dot(X.T, de_dz)
            self.grad_b[0] = np.sum(de_dz, axis=0)
        # L2-regulization grad
        for i in range(self.n_layers + 1):
            self.grad_W[i] += self.alpha * self.W[i]

    def __update_params(self):

        # 1. 随机梯度下降法，一般不用
        if self.solver == 'sgd':
            # NAD梯度优化
            for i in range(self.n_layers + 1):
                self.velocities_W[i] = self.momentum * self.velocities_W[i] - self.learning_rate * self.grad_W[i]
                self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * self.grad_b[i]

                self.W[i] += self.velocities_W[i]
                self.b[i] += self.velocities_b[i]
        elif self.solver == 'adam':
            # adam梯度优化方法
            self.iter_count += 1
            for i in range(self.n_layers + 1):
                self.mt_W[i] = self.beta1 * self.mt_W[i] + (1 - self.beta1) * self.grad_W[i]
                self.mt_b[i] = self.beta1 * self.mt_b[i] + (1 - self.beta1) * self.grad_b[i]
                self.vt_W[i] = self.beta2 * self.vt_W[i] + (1 - self.beta2) * (self.grad_W[i] ** 2)
                self.vt_b[i] = self.beta2 * self.vt_b[i] + (1 - self.beta2) * (self.grad_b[i] ** 2)

                learning_rate = (self.learning_rate * np.sqrt(1 - self.beta2 ** self.iter_count) /
                                 (1 - self.beta1 ** self.iter_count))
                self.W[i] += -learning_rate * self.mt_W[i] / (np.sqrt(self.vt_W[i]) + self.epsilon)
                self.b[i] += -learning_rate * self.mt_b[i] / (np.sqrt(self.vt_b[i]) + self.epsilon)


    def __compute_loss(self, x, y):

        # 对数似然函数 loss = ylog(x + 1e-8)
        prob = np.clip(x, 1e-10, 1.0 - 1e-10)
        loss = -np.sum(y * np.log(prob))
        # L2正则化损失
        for i in range(self.n_layers + 1):
            loss += self.alpha * (np.sum(self.W[i] * self.W[i]) / 2.0)
        return loss

    def __Check_X_y(self, X, y):
        # 检查数据格式
        if not X.ndim == 2:
            nfeatures = X.shape[0]
            X = X.reshape(1, nfeatures)
        if not y.ndim == 2:
            noutputs = y.shape[0]
            y = y.reshape(1, noutputs)
        return X, y

if __name__=='__main__':
    import sys
    sys.path.append('..')
    from preprocessing.data import load_reduction_data

    X_train, X_val, X_test, y_train, y_val, y_test = load_reduction_data('r50')
    model = NeuralNet(hidden_layers=(100,), batch_size=200, learning_rate=1e-3,
                     max_iter=100, tol=1e-4, alpha=1e-5,
                     activation='relu', solver='sgd')
    out_num = np.max(y_train)+1
    train_num = len(y_train)
    val_num = len(y_val)
    y_train_sparse = np.zeros((train_num, out_num))
    y_train_sparse[np.arange(0, train_num), y_train] = 1
    y_val_sparse = np.zeros((val_num, out_num))
    y_val_sparse[np.arange(0, val_num), y_val] = 1
    model.fit(X_train, y_train_sparse)
    accurancy_curve = model.acc_curve
    plt.plot(accurancy_curve)
    plt.show()
    y_pred = model.predict(X_val)
    print('accuracy:', 1 - np.sum(np.abs(y_pred - y_val_sparse)) / 2 / y_val_sparse.shape[0])