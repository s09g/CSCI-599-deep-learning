import numpy as np


""" Super Class """
class Optimizer(object):
	""" 
	This is a template for implementing the classes of optimizers
	"""
	def __init__(self, net, lr=1e-4):
		self.net = net  # the model
		self.lr = lr    # learning rate

	""" Make a step and update all parameters """
	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				pass


""" Classes """
class SGD(Optimizer):
	""" Some comments """
	def __init__(self, net, lr=1e-4):
		self.net = net
		self.lr = lr

	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dv = layer.grads[n]
				layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
	def __init__(self, net, lr=1e-4, momentum=0.0):
		self.net = net
		self.lr = lr
		self.momentum = momentum
		self.velocity = {}

	def step(self):
		#############################################################################
		#  Implement the SGD + Momentum                                        #
		#############################################################################
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				velocity = self.velocity.get(n, 0)
				velocity = self.momentum * velocity - self.lr * layer.grads[n]
				layer.params[n] += velocity
				self.velocity[n] = velocity
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class RMSProp(Optimizer):
	def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
		self.net = net
		self.lr = lr
		self.decay = decay
		self.eps = eps
		self.cache = {}  # decaying average of past squared gradients

	def step(self):
		#############################################################################
		#  Implement the RMSProp                                               #
		#############################################################################
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				grad = layer.grads[n]
				decaying_average = self.decay * self.cache.get(n, 0) + (1 - self.decay) * (grad ** 2)
				self.cache[n] = decaying_average
				layer.params[n] -= (self.lr * grad) / np.sqrt(decaying_average + self.eps)

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class Adam(Optimizer):
	def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
		self.net = net
		self.lr = lr
		self.beta1, self.beta2 = beta1, beta2
		self.eps = eps
		self.mt = {}
		self.vt = {}
		self.t = t

	def step(self):
		#############################################################################
		# Implement the Adam                                                  #
		#############################################################################
		self.t = self.t + 1
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				grad = layer.grads[n]
				m = self.beta1 * self.mt.get(n, 0) + (1 - self.beta1) * grad
				self.mt[n] = m
				mt = m / (1 - (self.beta1 ** self.t))
				v = self.beta2 * self.vt.get(n, 0) + (1 - self.beta2) * (grad ** 2)
				self.vt[n] = v
				vt = v / (1 - (self.beta2 ** self.t))
				layer.params[n] -= (self.lr * mt) / (np.sqrt(vt) + self.eps)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################