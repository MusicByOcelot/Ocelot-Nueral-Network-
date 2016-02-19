# Created by Marcello Martins on Feb 14, 2016
# Last modified on Feb 15, 2016
# Create NN and teach it with data collected from @file data.py

# @file NN.py

import numpy as np
import neurolab as nl


# function to quickly display the errors for testingData
def missRate(ary, testingGenreOrder):
	t = (0,1,2,3)
	miss = [0,0,0,0,0]
	for x in t:
		l = list(t)
		l.remove(x)
		for ele in ary[x*20:x*20+20]:
			for i in l:
				if ele[x] <= ele[i]:
					miss[x]+=1
					miss[4]+=1
					break
	print("{} miss rate: {}%".format(testingGenreOrder[0], miss[0]/20.0*100))
	print("{} miss rate: {}%".format(testingGenreOrder[1], miss[1]/20.0*100))
	print("{} miss rate: {}%".format(testingGenreOrder[2], miss[2]/20.0*100))
	print("{} miss rate: {}%".format(testingGenreOrder[3], miss[3]/20.0*100))
	print("Total miss rate: {}%".format(miss[4]/80.0*100))

def run():
	# load data from storage
	print("Loading Data from storage...")
	learningGenreOrder = np.load("../storage/learningGenreOrder12.npy")
	testingGenreOrder = np.load("../storage/testingGenreOrder12.npy")
	learningData = np.load("../storage/learningData12.npy")
	answersData = np.load("../storage/answersData12.npy")
	testingData = np.load("../storage/testingData12.npy")

	# create network with 7 inputs, 15 neurons in hidden layer and 4 in output layer
	# define that the range of inputs will be from -1 to 1 and there will be 
	print("Setting up NN...")
	net = nl.net.newff([[-1, 1]]*12, [15, 4])

	# train the NN
	print("Training NN...")
	err = net.train(learningData, answersData, show=100, epochs=1000, goal=0.01)

	# simulate the NN with the testing data
	print("Simulating NN...")
	ary = net.sim(testingData)

	# Display the miss rate for the testing data
	missRate(ary, testingGenreOrder)

	# save the NN for later use if needed
	save = raw_input("Would you like to save this NN? ").lower()
	if save == 'y' or save == 'yes':
		name = raw_input("Please enter file name to save NN: ")
	else:
		runAgain = raw_input("Run again? ").lower()
		if runAgain == 'y' or runAgain == 'yes':
			run()
		else:
			print("NN.py Completed")

run()