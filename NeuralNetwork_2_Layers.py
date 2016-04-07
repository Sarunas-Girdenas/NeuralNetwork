# this is simple Neural Network implementation, courtesy to http://iamtrask.github.io/
'''
This is the simplest implementation of Neural Network with 'full batch' training
to improve the speed we can add the multiple batches
Bold Driver algo is implemented 
This is neural network with two layers.
'''

class NeuralNetwork_2_Layers(object):

	def __init__(self,weights_1=[],weights_2=[],doBoldDriver=True):

		# note, if we select BoldDriver=True, it will do BoldDriver for both layers!

		self.weights_1 = weights_1    # weights for layer 1
		self.weights_2 = weights_2    # weights for layer 1

		if doBoldDriver:
			self.changePos_1 = 0 # BoldDriver values for first layer
			self.changePos_2 = 0 # BoldDriver values for second layer
			self.changeNeg_1 = 0 # BoldDriver values for first layer
			self.changeNeg_2 = 0 # BoldDriver values for second layer
			self.valueBold_1 = 0 # BoldDriver values for first layer
			self.valueBold_2 = 0 # BoldDriver values for second layer
			self.doBoldDriver = True
		else:
			self.doBoldDriver = False

		return None

	def initializeBoldDriver(self,Layer,changePos,changeNeg,valueBold):
		'''
		Purpose: initialize bold driver if requested
		'''

		if self.doBoldDriver == False:
			raise Exception('Bold Driver cannot be initialized since it was NOT Requested!')

		if (changePos > 1) or (changePos < 0 ):
			raise Exception('Change Positive must be between 0 and 1!')
		elif (changeNeg > 1) or (changeNeg < 0):
			raise Exception('Change Negative must be between 0 and 1!')

		if Layer == 1:
			self.changePos_1 = changePos
			self.changeNeg_1 = changeNeg
			self.valueBold_1 = valueBold
		elif Layer == 2:
			self.changePos_2 = changePos
			self.changeNeg_2 = changeNeg
			self.valueBold_2 = valueBold

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

	def BoldDriver(self,learningRate,errorsInit,errors,Layer):
		'''
		Purpose: perform bold driver algorithm based on the size of error
		Input:   valueBold    - change of error
			     learningRate - current learning rate
			     errorsInit   - value of previous errors
			     errors       - value of current errors
			     Layer        - in which layer we use BoldDriver algorithm
		Output:  newLearningRate - updated learning rate
		'''

		if self.doBoldDriver == False:
			raise Exception('Bold Driver was not Requested, so it cannot be Called!')

		if Layer == 1:
			#actual bold driver algo
			if abs(errors) - abs(errorsInit) > self.valueBold_1:
				learningRate = self.changePos_1 * learningRate
			else:
				learningRate = (1 + self.changeNeg_1) * learningRate
			
		elif Layer == 2:
			if abs(errors) - abs(errorsInit) > self.valueBold_2:
				learningRate = self.changePos_2 * learningRate
			else:
				learningRate = (1 + self.changeNeg_2) * learningRate

		return learningRate


	def updateWeights(self,inputSet,inputLabels,maxIter,tolerance,learningRate_1,learningRate_2,batch,miniBatch):
		'''
		Purpose: do update using full batch gradient descent
		Input:   inputSet       - numpy matrix of inputs
			     inputLabels    - numpy array of labels
			     maxIter        - number of maximum iterations
			     tolerance      - tolerance of error
			     learningRate_1 - learning Rate for gradient descent, layer 1
			     learningRate_2 - learning Rate for gradient descent, layer 2
			     miniBatch      - True/False -> do miniBatch GD
			     batch 			- size of the mini batch
		Output:  storeErrors  - errors from predictions
		'''

		if type(batch) != int:
			raise Exception('Batch Size must be Integer!')

		if batch > len(inputSet):
			raise Exception('Batch cannot be Longer than Input Data!')

		from numpy import dot

		noIter = 0

		storeErrorsFinal = [] # store final errors of the network
		storeErrors_1    = [] # store errors from the middle (intermediate) layer

		errorsInit   = 10 # initial errors for final layer
		errors       = 1
		l1_errorInit = 1 # initial errors for the first layer

		if miniBatch:
			from numpy.random import choice
			batchIndex  = choice(len(inputSet),batch,replace=False)
			inputSet    = inputSet[batchIndex]
			inputLabels = inputLabels[batchIndex]

		# zero layer 
		l0 = inputSet

		while abs(errors) > abs(tolerance):

			noIter += 1

			# forward propagation through layers 0, 1 and 2
			
			l1 = NeuralNetwork_2_Layers.sigmoidActivation( dot(l0,self.weights_1) )
			l2 = NeuralNetwork_2_Layers.sigmoidActivation( dot(l1,self.weights_2) ) # input for second layer is output from the first

			error  = inputLabels - l2 # error of final layer (output)
			errors = sum(abs(error))

			l2_delta = error * NeuralNetwork_2_Layers.sigmoidActivation(l2,derivative=True)

			l1_error = l2_delta.dot(self.weights_2.T)

			l1_delta = l1_error * NeuralNetwork_2_Layers.sigmoidActivation(l1,derivative=True)

			# update weights with bold driver (if requested)

			if self.doBoldDriver == True:
				learningRate_1 = self.BoldDriver(learningRate_1,l1_errorInit,sum(sum(l1_error)),1)
				learningRate_2 = self.BoldDriver(learningRate_2,errorsInit,errors,1)
				self.weights_1 += learningRate_1 * l0.T.dot(l1_delta)
				self.weights_2 += learningRate_2 * l1.T.dot(l2_delta)
			elif self.doBoldDriver == False:
				self.weights_1 += learningRate_1 * l0.T.dot(l1_delta)
				self.weights_2 += learningRate_2 * l1.T.dot(l2_delta)

			if noIter == maxIter:
				print 'Algorithm Reached Number of Maximum Iterations!'
				break

			# by plotting those errors we can see if the network is working
			
			storeErrors_1.append(sum(abs(l1_error)))
			storeErrorsFinal.append(errors)

			errorsInit   = errors
			l1_errorInit = sum(sum(l1_error))

		return storeErrorsFinal, storeErrors_1

	def computePredictionNN(self,inputData):
		'''
		Purpose: compute prediction based on the training
		Input:   inputData      - numpy array where columns are features and rows are observations
		Output:	 predictionsOut - numpy array of predictions, the same length as inputData
		'''
		
		from numpy import dot

		l0 = inputData
		l1 = NeuralNetwork_2_Layers.sigmoidActivation( dot(l0,self.weights_1) )
		l2 = NeuralNetwork_2_Layers.sigmoidActivation( dot(l1,self.weights_2) )

		predictionsOut = l2

		return predictionsOut