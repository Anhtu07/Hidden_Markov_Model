 #!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np
from hmmlearn import hmm
import librosa
import os
from sklearn.manifold import TSNE 
from matplotlib import pyplot as plt
import soundfile as sf

class DataLoader():
    def __init__(self, path, n_mfcc, window_size, overlap, shuffle = True, test_size = 0.2, stupid_mode = False):
        self.fpaths = {}
        self.spoken = []
        self.test_size = test_size
        self.stupid_mode = stupid_mode
        self.path = path
        self.shuffle = shuffle
        
        self.train_features = {}
        self.train_lengths = {}
        self.test_features = []
        self.test_labels = []
        
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.overlap = overlap
        
    def load_data(self):
        fea_ar = []
        lab_ar = []
        # Tutturu, load path
        for f in os.listdir(self.path):
            self.fpaths[f] = []
            # self.features[f] = np.zeros((0, n_mfcc))
            # self.f_lengths[f] = []
            for w in os.listdir(self.path + '/' + f):
                # print('audio/' + f + '/' + w)
                self.fpaths[f].append(self.path + '/' + f + '/' + w)
                if f not in self.spoken:
                    self.spoken.append(f)
        
        # Tutturu, load wav
        for word in self.spoken:
            self.word_path = self.fpaths[word]
            self.train_features[word] = np.zeros((0, self.n_mfcc))
            self.train_lengths[word] = []
            current_features = []
            current_labels = []
            for f in self.word_path:
                d, sr = sf.read(f, dtype='float32')
                d = d.T
                d = librosa.resample(d, sr, 22050)
                # d, sr = librosa.load(f)
                # Shape: (t, n_mfcc_dim)s
                feature = librosa.feature.mfcc(d, n_fft = int(sr*self.window_size), hop_length = int(sr*self.overlap), n_mfcc=self.n_mfcc).T
                # self.features[word] = np.concatenate((features[word], feature))
                # self.f_lengths[word].append(len(feature))
                current_features.append(feature)
                current_labels.append(word)
            
            if self.shuffle:
                np.random.shuffle(current_features)
                
            test_idx = int((1-self.test_size)*len(current_features))
            current_train_features = current_features[0:test_idx]
            current_train_labels = current_labels[0:test_idx]
            fea_ar.extend(current_train_features)
            lab_ar.extend(current_train_labels)
            if not self.stupid_mode:
                self.test_features.extend(current_features[test_idx:])
                self.test_labels.extend(current_labels[test_idx:])
            else:
                self.test_features.extend(current_features)
                self.test_labels.extend(current_labels)
        
        for idx, feature in enumerate(fea_ar):
            lab = lab_ar[idx]
            self.train_features[lab] = np.concatenate((self.train_features[lab], feature))
            self.train_lengths[lab].append(len(feature))
            
            
        

class SpeechModel():
    def __init__(self, data_loader, n_com, n_mix):
        self.classes = []
        self.models = {}
        self.n_com = n_com
        self.n_mix = n_mix
        self.data_loader = data_loader
        
    
    def fit(self):
        # data_loader: DataLoader Class
        start, trans = initByBakis(self.n_com, 2)
        for word in self.data_loader.spoken:
            self.classes.append(word)
            self.models[word] = hmm.GMMHMM(n_components=self.n_com, n_mix = self.n_mix, covariance_type="diag", n_iter = 500)
            self.models[word].transmat_ = trans
            self.models[word].startprob_ = start
            self.models[word].fit(self.data_loader.train_features[word], self.data_loader.train_lengths[word])
    
    def get_class(self, feature):
        scores = []
        for key, model in self.models.items():
            scores.append(model.score(feature))
        return list(self.models.keys())[np.argmax(scores)]
    
    def cal_accuracy(self):
        total = len(self.data_loader.test_features)
        print("Train data size: " + str(sum(map(lambda x: len(x), self.data_loader.train_lengths.values()))))
        print("Test data size: " + str(total))
        total_true = 0
        for idx in range(total):
            if self.get_class(self.data_loader.test_features[idx]) == self.data_loader.test_labels[idx]:  
                total_true = total_true + 1
        return total_true/total
    
    def test_from_audio(self, path):
        d, sr = sf.read(path, dtype='float32')
        d = d.T
        d = librosa.resample(d, sr, 22050)
        feature = librosa.feature.mfcc(d, n_fft = int(sr*self.data_loader.window_size), hop_length=int(sr*self.data_loader.overlap), 
                                       n_mfcc=self.data_loader.n_mfcc).T
        for key, model in self.models.items():
            print(key + "\t:" + str(model.score(feature)))
        return self.get_class(feature)
    
def initByBakis(inumstates, ibakisLevel):
    startprobPrior = np.zeros(inumstates)
    startprobPrior[0: ibakisLevel - 1] = 1/float((ibakisLevel - 1))
    transmatPrior = getTransmatPrior(inumstates)
    return startprobPrior, transmatPrior

def getTransmatPrior(inumstates):
    transmat = np.zeros((inumstates, inumstates))
    for i in range(inumstates):
        if i == inumstates-1:
            transmat[i, i] = 1.0
        else:
            transmat[i, i] = 0.5
            transmat[i, i+1] = 0.5
    return transmat

if __name__ == '__main__':
    data_loader = DataLoader(path='tiengviet', n_mfcc=6, window_size=0.015,
                             overlap=0.005, test_size = 0.2, shuffle=True)
    data_loader.load_data()
    model = SpeechModel(data_loader, n_com=3, n_mix=5)
    model.fit()
    print("Accuracy: " + str(model.cal_accuracy()))