# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:23:23 2021

@author: LG
"""

import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

# In[1]
file = 'AnyConv.com__000002.wav'

sig, sr = librosa.load(file, sr = 22050)

#print(sig,sig.shape)

plt.figure(figsize = FIG_SIZE)
librosa.display.waveplot(sig, sr, alpha = 0.5)

# In[2]
hop_length = 512
n_fft = 2048

hop_length_duration = float(hop_length)/sr
n_fft_duration = float(n_fft)/sr

stft = librosa.stft(sig, n_fft = n_fft, hop_length = hop_length)

magnitude = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(magnitude)

plt.figure(figsize = FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr = sr, hop_length=hop_length)
# In[3]
print(sig,sig.shape)
print(log_spectrogram, log_spectrogram.shape)