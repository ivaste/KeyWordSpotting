"""
	Load and preprocess Dataset

Dataset files must be already in .npy format

"""

import numpy as np
import librosa 		# mfcc functions
import pandas as pd # reading txt files
import random 		# to shuffle dataset
import os

sr=16000	#Sampling rate of audiofiles

diCategories = {
            'yes': 0,
            'no': 1,
            'up': 2,
            'down': 3,
            'left': 4,
            'right': 5,
            'on': 6,
            'off': 7,
            'stop': 8,
            'go': 9}
			
basePath = 'sd_GSCmdV2'

def loadDatasetFilenames(nCategories=10):
	global diCategories
	if nCategories == 21:
		diCategories = {
			'yes': 0,
            'no': 1,
            'up': 2,
            'down': 3,
            'left': 4,
            'right': 5,
            'on': 6,
            'off': 7,
            'stop': 8,
            'go': 9,
			'zero': 10,
			'one': 11,
			'two': 12,
			'three': 13,
			'four': 14,
			'five': 15,
			'six': 16,
			'seven': 17,
			'eight': 18,
			'nine': 19,
			'unknown': 20}
	else:
		diCategories = {
            'yes': 0,
            'no': 1,
            'up': 2,
            'down': 3,
            'left': 4,
            'right': 5,
            'on': 6,
            'off': 7,
            'stop': 8,
            'go': 9}
	
	categoriesFolder=tuple([e+"/" for e in diCategories])
	categories=tuple(diCategories)

	testWAVs = pd.read_csv(basePath + '/train/testing_list.txt',
							   sep=" ", header=None)[0].tolist()
	valWAVs = pd.read_csv(basePath + '/train/validation_list.txt',
								   sep=" ", header=None)[0].tolist()
								   
	if nCategories==21:
		testWAVs = [os.path.join(basePath + '/train/', f + '.npy')
					for f in testWAVs if f.endswith('.wav')]
		valWAVs = [os.path.join(basePath + '/train/', f + '.npy')
						for f in valWAVs if f.endswith('.wav')]
	else:
		testWAVs = [os.path.join(basePath + '/train/', f + '.npy')
					for f in testWAVs if f.endswith('.wav') and f.startswith(categoriesFolder)]
		valWAVs = [os.path.join(basePath + '/train/', f + '.npy')
						for f in valWAVs if f.endswith('.wav') and f.startswith(categoriesFolder)]

	allWAVs = []
	for root, dirs, files in os.walk(basePath + '/train/'):
		if nCategories==21:
			allWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy')]
		else:
			allWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy') and root.endswith(categories)]
	trainWAVs = list(set(allWAVs) - set(valWAVs) - set(testWAVs))
	
	#shuffle lists
	random.shuffle(trainWAVs)
	random.shuffle(valWAVs)
	random.shuffle(testWAVs)
	
	#print("# of test: ",len(testWAVs))
	#print("# of val: ",len(valWAVs))
	#print("# of train: ",len(trainWAVs))
	#print("# total: ",len(allWAVs))
	return trainWAVs,valWAVs,testWAVs


# LOAD DATASET FILES
def loadBatch(filesList,batch_size=1000,dim=16000,nCategories=10):

	X = np.empty((batch_size, dim))
	y = np.empty((batch_size), dtype=int)

    # Generate data
	for i, ID in enumerate(filesList[0:batch_size]):
		# load data from file, saved as numpy array on disk
		curX = np.load(ID)

		# curX could be bigger or smaller than self.dim
		if curX.shape[0] == dim:
			X[i] = curX
		elif curX.shape[0] > dim:  # bigger
			# we can choose any position in curX-self.dim
			randPos = np.random.randint(curX.shape[0]-dim)
			X[i] = curX[randPos:randPos+dim]
		else:  # smaller
			randPos = np.random.randint(dim-curX.shape[0])
			X[i, randPos:randPos + curX.shape[0]] = curX
			# print('File dim smaller')
		
		# Store class
		if nCategories==21:
			if os.path.basename(os.path.dirname(ID)) not in diCategories:
				y[i]=20#Unknown
			else:
				y[i] = diCategories[os.path.basename(os.path.dirname(ID))]
		else:
			y[i] = diCategories[os.path.basename(os.path.dirname(ID))]
        
	return X,y


###################################################################
################ PREPROCESSING ####################################
###################################################################

#12 MFCC + DELTA + DELTADELTA
def MFCC_DELTA(X,n_mfcc=12,sr=16000): #X: (n_examples,...) 
    features = np.empty((X.shape[0],n_mfcc*3,126)) #12*3, ...
    for i,y in enumerate(X):
        S = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024,
                                                hop_length=128, power=1.0, #window='hann',
                                                n_mels=80, fmin=40.0, fmax=sr/2)

        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)

        # Next, we'll extract the top 12 Mel-frequency cepstral coefficients (MFCCs)
        mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)

        # Let's pad on the first and second deltas while we're at it
        delta_mfcc  = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        features[i] = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
    return features



def MFCC(X,n_mfcc=12,sr=16000):
	features = np.empty((X.shape[0],n_mfcc,126))
	for i,y in enumerate(X):
		S = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024,
										hop_length=128, power=1.0,
										n_mels=80, fmin=40.0, fmax=sr/2)

		# Convert to log scale (dB). We'll use the peak power (max) as reference.
		log_S = librosa.power_to_db(S, ref=np.max)

		# Next, we'll extract the top n_mfcc Mel-frequency cepstral coefficients (MFCCs)
		features[i]= librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)

	return features
	
	

def melspect(X,nMels=80,sr=16000):
	features = np.empty((X.shape[0],nMels,126))	#nExamples, nMels, n???
	for i,y in enumerate(X):
		S = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024,
										hop_length=128, power=1.0,
										n_mels=nMels, fmin=40.0, fmax=sr/2)
										
		# Convert to log scale (dB). We'll use the peak power (max) as reference.
		features[i]=librosa.power_to_db(S, ref=np.max)
	
	return features
	
