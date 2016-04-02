# this is simple Neural Network implementation, courtesy to http://iamtrask.github.io/
'''
This is the simplest implementation of Neural Network with 'full batch' training
to improve the speed we can add the multiple batches and bold diver algorithm 

'''

class NeuralNetworkBatch(object):

	def __init__(self,weights=[]):

		self.weights = weights

		return None

	@staticmethod
	def sigmoidActivation(x,derivative=False):

		from numpy import exp
		'''
		Purpose: return value of a sigmoid function and also return derivativate if requested
		Input:   x - matrix/array
		Outpu:   sigmoid values
		'''

		if derivative == True:
			return x*(1-x)

		sigmoidOut = (1 / (1+exp(-x)))

		return sigmoidOut

	def initializeWeights(self,numOfWeights):
		'''
		Purpose: initialize weights
		Input:	 number of weights
		Output:  numpy array
		NOTE: in this case weights are initialized randomly!
		'''

		from numpy.random import random

		self.weights = 2*random((numOfWeights,1)) - 1 

		return None

	def updateWeights(self,inputSet,inputLabels,maxIter,tolerance,learningRate):
		'''
		Purpose: do update using full batch gradient descent
		Input:   inputSet     - numpy matrix of inputs
			     inputLabels  - numpy array of labels
			     maxIter      - number of maximum iterations
			     tolerance    - tolerance of error
			     learningRate - learning Rate for gradient descent
		Output:  storeErrors  - errors from predictions
		'''

		from numpy import dot

		noIter = 0

		errors = 0.1

		storeErrors = []

		while abs(errors) > abs(tolerance):

			noIter += 1

			# forward propagation

			l0 = inputSet
			l1 = NeuralNetworkBatch.sigmoidActivation( dot(inputSet,self.weights) )

			error = inputLabels - l1
			errors = sum(abs(error))

			l1_delta = error * NeuralNetworkBatch.sigmoidActivation(l1,derivative=True)

			# update weights

			self.weights += learningRate * dot(inputSet.T,l1_delta)

			if noIter == maxIter:
				print 'Algorithm Reached Number of Maximum Iterations!'
				break

			# by plotting those errors we can see if the 

			storeErrors.append(errors)

		return storeErrors


