import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# This is just a handy-dandy function I made a while back to make plotting graphs easier
def labelGraph(title, x_axis, y_axis):
	plt.suptitle(title, fontsize=20)
	plt.xlabel(x_axis, fontsize=16)
	plt.ylabel(y_axis, fontsize=16)


# Lets make a logistic regression class to house all the related functions
class logisticRegression:

	# Basic constructor to define the learning rate and iterations
	def __init__(self, learningRate, numIter):
	  self.learningRate = learningRate
	  self.numIter = numIter

	# Just a 'clean' way to get the sigmoid result
	def logisticFunction(self, z):
		return 1 / (1 + np.exp(-z))

	# Now we are going to use gradient descent to derive the binary classification line
	def gradientDescent(self, trainData, targetData):

		# Adding a row of '1s' to be the y-intercept of the function
	  trainData = np.concatenate((np.ones((trainData.shape[0], 1)), trainData), axis=1)
	  
	  # This is just a place to store the betas for the regression
	  self.betas = np.zeros(trainData.shape[1])
	  
	  for i in range(self.numIter):
	  		# We take the dot product to get the value which we exponentiate in the sigmoid function
	      result = np.dot(trainData, self.betas)

	      # Then we get the hypothesis value by evaluating the logistic function with the result
	      outcome = self.logisticFunction(result)

	      # Now this is where the algorithm does its 'learning' and adjusts the betas by the effect they have on the result
	      gradientFactor = np.dot(trainData.T, (outcome - targetData)) / len(targetData)
	      self.betas -= (self.learningRate * gradientFactor)

# This is our input file used for training the machine
fileName = 'trainingFile.txt'

test = open(fileName, 'r')
fileData = []

# We are just going to store the information in a pandas dataframe for easy manipulation 
for line in test:
	line = line.replace('\n', '').split('\t')

	if len(line) != 1:
		for i in range(len(line)):
			if i == 2:
				line[i] = int(float(line[i]))
			else:
				line[i] = float(line[i])

		fileData.append(line)
	else:
		continue

dataFrame = pd.DataFrame(fileData)

training_data = dataFrame.iloc[:,0:2]
training_target = dataFrame.iloc[:,2]

# Lets save a copy of our dataframe for later on
plt.scatter(dataFrame[0], dataFrame[1], c = dataFrame[2])
plt.ylim(0,5)
plt.xlim(-1,5)

model = logisticRegression(0.01, 15000)
model.gradientDescent(training_data, training_target)
print('Beta0 =', f'{model.betas[0]:.7f}','\nBeta1 =', f'{model.betas[1]:.7f}', '\nBeta2 =', f'{model.betas[2]:.7f}')

plot_x = np.array([min(dataFrame.iloc[:,0]) - 3, max(dataFrame.iloc[:,0]) + 2])
plot_y = (-1/model.betas[2]) * (model.betas[1] * plot_x + model.betas[0])
print('Points on linear decision boundry:', '(' + str(plot_x[0]) + ',' + str(round(plot_y[0], 4)) + ') and ' + '(' + str(plot_x[1]) + ',' + str(round(plot_y[1],4)) + ')')

plt.plot(plot_x, plot_y)
labelGraph('Logistic Classifier', 'X-Points', 'Y-Points')
plt.show()

modelProbability = model.logisticFunction(-np.dot([1,3,4], model.betas))
print("The probability the point (3,4) is in class '1' is", f'{modelProbability:.4f}')
