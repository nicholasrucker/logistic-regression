import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import itertools
import numpy as np
import math

# Cost function is essentially an adjusted total sum of sqaures statistic
def costFunction(x_features, y_outcome, betas):
	SST = 0
	for j in range(len(x_features)):
		outcome = -1 * ((x_features[j][0] * betas[0]) + (x_features[j][1] * betas[1]) + (x_features[j][2] * betas[2]))
		outcome2 = 1 / (1 + math.e**outcome)
		y_outcome[j] = 1 if y_outcome[j] > 0 else 0
		SST += y_outcome[j] * math.log(outcome2) + ((1 - y_outcome[j]) * math.log(1 - outcome2))

	return SST / (-len(x_features))

# This is where the magic happens
# We will need a list of x_features and y_outcomes for trainings
# Betas is a list of our betas! (eg: the effect of x on y)
# Iterations is how many times we want to refit the line
# Lastly, alpha is our learning rate.  This is something that can be changed around!
# Also, a list of all the costs for each iterations will be returned for easy graphing
def gradientD(x_features, y_outcome, betas, iterations, alpha):
	cost = np.zeros(iterations)

	for i in range(iterations):
		eval0 = 0
		eval1 = 0
		eval2 = 0
		for j in range(len(x_features)):

			eval0 = x_features[j][0] * (1/(1 + (math.e**-((x_features[j][0] * betas[0]) + (x_features[j][1] * betas[1]) + (x_features[j][2] * betas[2]))) - y_outcome[j]))
			eval1 = x_features[j][1] * (1/(1 + (math.e**-((x_features[j][0] * betas[0]) + (x_features[j][1] * betas[1]) + (x_features[j][2] * betas[2]))) - y_outcome[j]))
			eval2 = x_features[j][2] * (1/(1 + (math.e**-((x_features[j][0] * betas[0]) + (x_features[j][1] * betas[1]) + (x_features[j][2] * betas[2]))) - y_outcome[j]))
		
		betas[0] = betas[0] - (alpha * eval0)
		betas[1] = betas[1] - (alpha * eval1)
		betas[2] = betas[2] - (alpha * eval2)

		cost[i] = costFunction(x_features, y_outcome, betas) 
	return betas, cost

# This is just a handy-dandy function I made a while back to make plotting graphs easier
def labelGraph(title, x_axis, y_axis):
	plt.suptitle(title, fontsize=20)
	plt.xlabel(x_axis, fontsize=16)

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

# Now we should split the data up so we can evaluate our hypothesis
# For this example we will use 70% (210 observations) to train and 30% (90 observations) to test
randomSlice = np.random.rand(len(dataFrame)) < 0.8
train = dataFrame[randomSlice]
test = dataFrame[~randomSlice]

# Lets isolate our train and test features
dataTrain = train.iloc[:,0:2]
dataTest = test.iloc[:,0:2]

# NumPy has nice built in functionality to create a list of length '1' and the depth of the dataframe 
# So lets create two of those bad boys and join them to their respective data frame
testOnes = np.ones([dataTest.shape[0], 1])
dataTest = np.concatenate((testOnes, dataTest), axis = 1)

trainOnes = np.ones([dataTrain.shape[0], 1])
dataTrain = np.concatenate((trainOnes, dataTrain), axis = 1)

targetTrain = train.iloc[:,2].values
# Isolating target values
targetTest = test.iloc[:,2].values

# The numbers I chose to start off with were determined after running the algorithm a few times 
# Basically hyperparameter optimization (its a lot less fancy than it sounds)
betas = [1, 1, 1]

# A learning rate of .1 seemed to reduce the cost at a significant rate
alpha = .05
iterations = 10000

lastBetas, costs = gradientD(dataTrain, targetTrain, betas, iterations, alpha) 

print(lastBetas)
plot_x = np.array([min(dataTrain[:,0]) - 3, max(dataTrain[:,0]) + 2])
plot_y = (-1/lastBetas[2]) * (lastBetas[1] * plot_x + lastBetas[0])

plt.scatter(train[0], train[1], c = train[2], cmap = 'viridis')
plt.plot(plot_x, plot_y)
labelGraph("Title", "Body Length (cm)", "Dorsal Fin Length (cm)")
plt.show()

while True:

	print("\nEnter a value for body length and dorsal fin length (press enter after each entry): ")
	firstValue, secondValue = float(input()), float(input())
	
	if firstValue == 0 and secondValue == 0:
		break

	firstValue = (firstValue - savedDF[0].mean()) / savedDF[0].std()
	secondValue = (secondValue - savedDF[1].mean()) / savedDF[1].std()

	outcome = -1 * (lastBetas[0] + (lastBetas[1] * firstValue) + (lastBetas[2] * secondValue))
	expectedFish = 1 / (1 + math.e**outcome)

	expectedFish = '1' if (expectedFish) > .5 else '0'
	print("Expected to be: TigerFish" + expectedFish)