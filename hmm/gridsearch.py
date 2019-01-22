import numpy as np
import itertools
import model

n_coms = np.arange(3, 6)
n_mixs = np.arange(3, 6)
window_sizes = np.arange(0.015, 0.045, 0.01)
strides = np.arange(0.005, 0.015, 0.005)
n_mfccs = np.array([6, 9, 13])

params = itertools.product(n_coms, n_mixs, window_sizes, strides, n_mfccs)
max_setting = {}
max_acc = -np.inf

for n_com, n_mix, window_size, stride, n_mfcc in params:
    print('______')
    print('n_com = {n_com}, n_mix = {n_mix}, window_size = {window_size}, stride = {stride}, \
n_mfcc = {n_mfcc}'.format(n_com=n_com, n_mix=n_mix, window_size=window_size, stride=stride, n_mfcc=n_mfcc))
    data_loader = model.DataLoader(path='tiengviet', n_mfcc=n_mfcc, window_size=window_size,
                             overlap=stride, test_size=0.1 , shuffle = True)
    data_loader.load_data()
    m = model.SpeechModel(data_loader, n_com=n_com, n_mix=n_mix)
    m.fit()
    acc = m.cal_accuracy()
    print(acc)
    if acc > max_acc:
        max_acc = acc
        max_setting['n_com'] = n_com
        max_setting['n_mix'] = n_mix
        max_setting['window_size'] = window_size
        max_setting['stride'] = stride
        max_setting['n_mfcc'] = n_mfcc

