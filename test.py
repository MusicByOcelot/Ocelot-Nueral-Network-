import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
import wave as wav

wavFileRead = wav.open('test.wav', 'r')

def sampleNext(n):
	wavFileSamp = wav.open('samp.wav', 'w')
	sample = wavFileRead.readframes(n)
	params = list(wavFileRead.getparams())
	params[3] = n
	wavFileSamp.setparams(tuple(params))
	wavFileSamp.writeframes(sample)
	wavFileSamp.close()
	rate, data = wavfile.read('samp.wav') # load the data
	a = data.T[0] # this is a two channel soundtrack, I get the first track
	b = [(ele/2**32.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
	c = fft(b) # calculate fourier transform (complex numbers list)
	d = len(c)/2  # you only need half of the fft list (real signal symmetry)
	return rate, data


