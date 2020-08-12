#Imports
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import Models #Our models
import LoadAndPreprocessDataset
#.......................

categories=['yes','no','up','down','left','right','on','off','stop','go']
#categories=['yes','no','up','down','left','right','on','off','stop','go','zero','one','two','three','four','five','six','seven','eight','nine','unknown']


sr = 16000
seconds = 10

l=[]

def print_sound(indata, frames, time, status):
    
    #print(indata)
    #print(len(indata))
    l.append(indata)
    

with sd.InputStream(samplerate=sr, channels=1,callback=print_sound):
    sd.sleep(1000)

import time
time.sleep(2)
print(len(l))
print(l[0])
print("\n\n\n\n\n")
print(l[1])

