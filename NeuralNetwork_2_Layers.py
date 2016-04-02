# this is simple Neural Network implementation, courtesy to http://iamtrask.github.io/
'''
This is the simplest implementation of Neural Network with 'full batch' training
to improve the speed we can add the multiple batches and bold diver algorithm 
This is neural network with two layers.
'''

class NeuralNetwork_2_Layers(object):

	def __init__(self,weights_1=[],weights_2=[]):

		self.weights_1 = weights_1 # weights for layer 1
		self.weights_2 = weights_2 # weights for layer 1

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

	def initializeWeights_Layer1(self,L1_dim1,L1_dim2):
		'''
		Purpose: initialize weights in Layer 1
		Input:	 L1_dim1 - first layer rows
				 L2_dim2 - first layer columns
		Output:  numpy array
		NOTE: in this case weights are initialized randomly!
		'''

		from numpy.random import random

		self.weights_1 = 2*random((L1_dim1,L1_dim2)) - 1 

		return None

	def initializeWeights_Layer2(self,L1_dim2):
		'''
		Purpose: initialize weights in Layer 2
		Input:	 L1_dim2 - second layer columns
		Output:  numpy array
		NOTE: in this case weights are initialized randomly!
		NOTE: make sure that dimensions are matching!
		'''

		from numpy.random import random

		self.weights_2 = 2*random((L1_dim2,1)) - 1 # second dimension is 1 since we want just one output from our network!

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

		storeErrorsFinal = [] # store final errors of the network
		storeErrors_1    = [] # store errors from the middle (intermediate) layer

		while abs(errors) > abs(tolerance):

			noIter += 1

			# forward propagation through layers 0, 1 and 2

			l0 = inputSet
			l1 = NeuralNetwork_2_Layers.sigmoidActivation( dot(l0,self.weights_1) )
			l2 = NeuralNetwork_2_Layers.sigmoidActivation( dot(l1,self.weights_2) ) # input for second layer is output from the first

			error  = inputLabels - l2 # error of final layer (output)
			errors = sum(abs(error))

			l2_delta = error * NeuralNetwork_2_Layers.sigmoidActivation(l2,derivative=True)

			l1_error = l2_delta.dot(self.weights_2.T)

			l1_delta = l1_error * NeuralNetwork_2_Layers.sigmoidActivation(l1,derivative=True)

			# update weights

			self.weights_1 += l0.T.dot(l1_delta)
			self.weights_2 += l1.T.dot(l2_delta)

			if noIter == maxIter:
				print 'Algorithm Reached Number of Maximum Iterations!'
				break

			# by plotting those errors we can see if the network is working
			
			storeErrors_1.append(sum(l1_error))
			storeErrorsFinal.append(errors)

		return storeErrorsFinal, storeErrors_1


