import numpy as np
import os
import h5py
import pickle
import json
from scipy.fftpack import fft
import scipy.signal as signal
import matplotlib.pyplot as plt

def readBinFile(filename, packetLen):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    packetNum = int(len(data)/packetLen)
    data = data[:packetLen*packetNum]
    return data

def readSigMFFile(filename):
    with open('{}.sigmf-meta'.format(filename), 'rb') as f:
        meta_dict = json.load(f)
        sample_rate = meta_dict['_metadata']['global']['core:sample_rate']
        sample_start = meta_dict['_metadata']['captures'][0]['core:sample_start']
        sample_count = meta_dict['_metadata']['annotations'][0]['core:sample_count']
    with open('{}.sigmf-data'.format(filename), 'rb') as f:
        signal = np.fromfile(f, dtype=np.complex128)
        I = signal.real[sample_start:]
        Q = signal.imag[sample_start:]
    return I, Q, sample_rate, sample_count

def getFFT(complexSig, fs):
    fft_y = fft(complexSig)
    N = fs
    x = np.arange(N)  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    normalization_y = abs_y / N  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(N / 2))]  # 因为对称性，只取一半区间（单边频谱）
    return half_x, normalization_half_y


def showSignal(I, Q, fs):
    x = np.arange(len(I))
    plt.subplot(121)
    plt.plot(x, I, 'r')
    plt.plot(x, Q, 'b')
    plt.title('signal')

    plt.subplot(122)
    plt.plot(np.arange(256), I[:256], 'r')
    plt.plot(np.arange(256), Q[:256], 'b')
    plt.title('zoomed signal')

    sig = I + 1j * Q
    t, f = getFFT(sig, fs)
    plt.subplot(133)
    plt.plot(t, f)
    plt.show()

def showSignalSeg(I, Q, fs, segLen=256):
    plt.figure()
    plt.plot(np.arange(segLen), I[:segLen], 'r')
    plt.plot(np.arange(segLen), Q[:segLen], 'b')
    plt.title('Signal Segment')
    plt.show()


if __name__ == '__main__':
    filename = 'data/USRP310/WiFi_air_X310_3123D7B_2ft_run1'
    I, Q, fs, _ = readSigMFFile(filename)
    showSignalSeg(I, Q, fs)
    filename = 'data/USRP310/WiFi_air_X310_3123D7B_56ft_run1'
    I, Q, fs, _ = readSigMFFile(filename)
    showSignalSeg(I, Q, fs)
    print('test done!')