import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import librosa
import soundfile as sf
import librosa.display

d, sr = sf.read('D:/khongbietlacaiginua/tiengviet/mot/01.wav', dtype='float32')
d = d.T
d = librosa.resample(d, sr, 22050)
#Extract Raw Audio from Wav File
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveplot(d, sr=sr)
plt.title('Một')
mfccs = librosa.feature.mfcc(y=d, sr=sr, n_mfcc=6)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('Một')
plt.tight_layout()