# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 15:37 下午
# @Author  : Yijia Zheng
# @FileName: fast_smooth.py

'''
This file is about smoothed method included basic and fast solvers.
'''


class smooth_method():
	def __init__(self, A, b, mu, init_iteration, max_iteration, tol):
		self.A = A
		self.b = b
		self.m, self.n = self.A.shape
		self.mu = mu 
		self.init_iteration = init_iteration
		self.max_iteration = max_iteration
		self.tol = tol
		self.cov = np.dot(self.A.T, self.A)
		self.ATb = np.dot(self.A.T, self.b)
		self.step_size = 1.0/np.linalg.norm(self.cov, 2)
		self.result_path = []

	# define LASSO's object function
	def loss(self, w):
		w = w.reshape(-1)
		return 0.5 * np.sum(np.square(np.dot(self.A, w) - self.b)) + self.mu * np.sum(np.abs(w))

	# define the smoothed function--Huber Permalty
	def phi(self, z, t=1e-6):
		if np.abs(z) <= t:
			return z ** 2.0/(2.0 * t)
		else:
			return np.abs(z) - t / 2.0

	# define Huber Pernalty's proximal function.
	def prox_phi(self, x, alpha, t=1e-6):
		if np.abs(x) <= alpha+t:
			return 1.0 * x/(1.0 + alpha/t)
		elif x > alpha + t:
			return x - alpha
		else:
			return x + alpha

	# Huber Pernalty's gradient
	def deltaphi(self, z, t=1e-6):
		if np.abs(z) <= t:
			return 1.0 * z / t
		elif z > t:
			return 1.0
		else:
			return -1.0

	def train(self, method="BASIC"):
		'''
		Parameters
		----------
		method: string, 'BASIC'(default) or 'FISTA' or 'Nesterov'
				Specifies the method to train the model.
		'''
		import time
		start_time = time.time()
		print( method + ' is Solving...')
		self.phi = np.vectorize(self.phi)
		self.deltaphi = np.vectorize(self.deltaphi)
		self.prox_phi = np.vectorize(self.prox_phi)
		# initial weights
		self.x = np.random.normal(size=(self.n))
		self.x_ = self.x

		if method == "FISTA":
			def update(x, x_, k, mu):
				y = x + 1.0 * (k - 2)/(k + 1) * (x - x_)
				x_ = x[:]
				grad = np.dot(self.cov, y) - self.ATb
				tmp = y - self.step_size * grad
				x = self.prox_phi(tmp, alpha=self.step_size*mu)
				return x, x_

			for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
				for k in range(self.init_iteration):
					self.x, self.x_ = update(self.x, self.x_, k, hot_mu)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				self.x, self.x_ = update(self.x, self.x_, self.iters, mu)
				self.err_rate = np.abs(self.loss(self.x)-self.loss(self.x_))/self.loss(self.x_)
				self.iters += 1

		elif method == 'Nesterov':
			self.v = self.x[:]
			def update(x, v, k, mu):
				theta = 2.0/(k + 1)
				y = (1.0 - theta) * x + theta * v
				grad = np.dot(self.cov, y) - self.ATb + mu * self.deltaphi(y)
				tmp = v - self.step_size/theta * grad
				v = self.prox_phi(tmp, alpha=self.step_size*mu/theta)
				x = (1.0 - theta) * x + theta * v
				return x, v

			for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
				for k in range(self.init_iteration):
					self.x, self.v = update(self.x, self.v, k, hot_mu)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.x_ = self.x[:]
				self.result_path.append(self.loss(self.x))
				self.x, self.v = update(self.x, self.v, self.iters, mu)
				self.err_rate = np.abs(self.loss(self.x)-self.loss(self.x_))/self.loss(self.x_)
				self.iters += 1

		else:
			def update(x, mu):
				grad = np.dot(self.cov, x) - self.ATb + mu * self.deltaphi(x)
				x = x - self.step_size * grad
				return x

			for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
				for k in range(self.init_iteration):
					self.x = update(self.x, hot_mu)
					self.result_path.append(self.loss(self.x))

			self.iters = 1
			self.x_ = self.x[:]
			self.err_rate = 1.0
			while(self.err_rate > self.tol and self.iters < self.max_iteration):
				self.result_path.append(self.loss(self.x))
				self.x_ = self.x[:]
				self.x = update(self.x, self.mu)
				self.err_rate = np.abs(self.loss(self.x)-self.loss(self.x_))/self.loss(self.x_)
				self.iters += 1

		np.save('save/{}_smooth_w'.format(method), np.asarray(self.x))
		np.save('save/{}_smooth_err'.format(method), np.asarray(self.result_path))
		self.run_time = time.time() - start_time
		print ('End!')

	def plot(self, method='BASIC'):
		from bokeh.plotting import figure, output_file, show
		x = range(len(self.result_path))
		y = self.result_path
		output_file("./smooth"+method+".html")
		p = figure(title="Smooth Method_"+method, x_axis_label='iteratin', y_axis_label='loss')
		p.line(x, y, legend="smooth", line_width=2)
		show(p)


if __name__ == '__main__':

	import sys
	sys.path.append('..')
	from preprocessing.data import load_data
	import numpy as np
	from bokeh.plotting import figure, output_file, show

	# for reproducibility
	np.random.seed(2018111396)

	n = 1024
	m = 512
	mu = 1e-3
	init_iteration = int(1e2)
	max_iteration = int(1e3)
	tol = 1e-9

	X_train, X_val, X_test, y_train, y_val, y_test = load_data('r50')
	A = X_train
	b = y_train.reshape(-1)
	print(A.shape, b.shape)

	result_time = []
	output_file("Plots/smooth.html")
	p = figure(title="Smooth Method", x_axis_label='iteration', y_axis_label='loss')

	for method, color in zip(["BASIC", "FISTA", "Nesterov"], ["orange", "red", "blue"]):
		model = smooth_method(A, b, mu, init_iteration, max_iteration, tol)
		model.train(method)
		result_time.append(model.run_time)
		x = range(len(model.result_path))
		y = model.result_path
		p.line(x, y, legend_label=method, line_width=2, line_color=color)

	show(p)


