# Created by Marcello Martins on Feb 14, 2016
# Last modified on Feb 15, 2016
# Extract features/data from wav files and save to numpy array for testing

# @file data.py 

import librosa
import os
import numpy as np

# help function to get songs in directory while ignoring system files such as .DS_Store
def listdir_nohidden(path):
	redir = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			redir.append(f)
	return redir

# extract features/data from wav file and return as np.array
def getData(filename):
	print("Gretting data for{}".format(filename))
	hop_length = 256;

	# Load the example clip
	y, sr = librosa.load(filename)

	# Short-time Fourier transform (STFT)
	S = np.abs(librosa.stft(y))

	# Separate harmonics and percussives into two waveforms
	y_harmonic, y_percussive = librosa.effects.hpss(y)

	# Beat track on the percussive signal
	tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

	# Compute MFCC features from the raw signal
	mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

	# And the first-order differences (delta features)
	mfcc_delta = librosa.feature.delta(mfcc)

	# Stack and synchronize between beat events
	# This time, we'll use the mean value (default) instead of median
	beat_mfcc_delta = librosa.feature.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

	# Compute chroma features from the harmonic signal
	chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

	# Aggregate chroma features between beat events
	# We'll use the median value of each feature between beat frames
	beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)

	# Finally, stack all beat-synchronous features together
	beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

	# Average the energy 
	avgEnergy = np.mean(librosa.feature.rmse(y=y))

	# Estimate tuning
	tuning = librosa.estimate_tuning(y=y, sr=sr)

	#zeroCrossings = np.sum(librosa.core.zero_crossings(y=y))
	avgMelSpectro = np.mean(librosa.feature.melspectrogram(y=y, sr=sr))

	avgSpectralContrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr))

	raw = [ avgSpectralContrast, avgMelSpectro, np.mean(y_harmonic), np.mean(y_percussive), np.mean(mfcc), np.mean(mfcc_delta), np.mean(beat_mfcc_delta), np.mean(chromagram), np.mean(beat_chroma), np.mean(beat_features), avgEnergy, tuning]
	norm = [float(i)/sum(raw) for i in raw] # normalise numbers between -1 and 1
	return np.array([norm])

# fill array with content to teach NN
def fillLearningData(generes, genreOrder, numRange, data):
	for key, value in generes.iteritems():
		genreOrder = np.append(genreOrder,key)
		for files in value[0:numRange]:
			data = np.vstack([data,getData("../music/{}/{}".format(key, files))])
	return data, genreOrder

# fill array with content to test NN
def fillTestingData(generes, genreOrder, numRange, data):
	for key, value in generes.iteritems():
		genreOrder = np.append(genreOrder,key)
		for files in value[numRange:]:
			data = np.vstack([data,getData("../music/{}/{}".format(key, files))])
	return data, genreOrder

# create empty array to hold learningData set and testingDvata set
learningData = np.array([]).reshape(0,12)
testingData = np.array([]).reshape(0,12)

# create empty array to hold answersData for four generes
answersData = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
answersData = np.repeat(answersData, [80, 80, 80, 80], axis = 0)

# list of song names on my computer by genre
hiphop = listdir_nohidden("/Users/SimplyMarcello/Gdrive/Code/Python/Ocelot/music/hiphop")
jazz = listdir_nohidden("/Users/SimplyMarcello/Gdrive/Code/Python/Ocelot/music/jazz")
classical = listdir_nohidden("/Users/SimplyMarcello/Gdrive/Code/Python/Ocelot/music/classical")
country = listdir_nohidden("/Users/SimplyMarcello/Gdrive/Code/Python/Ocelot/music/country")

# create dictionary of genre titles and list of song
generes = {"hiphop" : hiphop, "jazz" : jazz, "classical" : classical, "country" : country}

# since dicionaries are stored in a random order we need to store
# the order in which they are read in the below arrays
learningGenreOrder = np.empty([0], dtype=str)
testingGenreOrder = np.empty([0], dtype=str)

# fill data sets with 80 songs for learning and 20 for testing
learningData, learningGenreOrder = fillLearningData(generes, learningGenreOrder, 80, learningData)
testingData, testingGenreOrder = fillTestingData(generes, testingGenreOrder, 80, testingData)

# save data for permanent storage
np.save("../storage/learningGenreOrder12.npy", learningGenreOrder)
np.save("../storage/testingGenreOrder12.npy", testingGenreOrder)
np.save("../storage/learningData12.npy", learningData)
np.save("../storage/answersData12.npy", answersData)
np.save("../storage/testingData12.npy", testingData)

print("data.py Completed")
