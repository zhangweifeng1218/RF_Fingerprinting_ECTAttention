import numpy as np
from scipy.fftpack import fft
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
mpl.rcParams['axes.unicode_minus']=False #显示负号

#采样点选择1400个，由于设置的信号频率份量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，因此这里设置采样频率为1400赫兹（即一秒内有1400个采样点，同样意思的）
fs = 1400
x=np.linspace(0,1,fs)
#设置须要采样的信号，频率份量有200，400和600
y=np.sin(2*np.pi*200*x) + 4*np.sin(2*np.pi*400*x)+6*np.sin(2*np.pi*600*x)
y=np.exp(1j*2*np.pi*200*x) + 7*np.exp(1j*2*np.pi*400*x)
#y = np.abs(y)
#y -= np.mean(y)
fft_y = fft(y)  # 快速傅里叶变换

N = 1400
x = np.arange(N)  # 频率个数
half_x = x[range(int(N / 2))]  # 取一半区间

abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
angle_y = np.angle(fft_y)  # 取复数的角度
normalization_y = abs_y / N  # 归一化处理（双边频谱）
normalization_half_y = normalization_y[range(int(N / 2))]  # 因为对称性，只取一半区间（单边频谱）

plt.subplot(231)
plt.plot(x, y)
plt.title('原始波形')

plt.subplot(232)
plt.plot(x, fft_y, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')

plt.subplot(233)
plt.plot(x, abs_y, 'r')
plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')

plt.subplot(234)
plt.plot(x, angle_y, 'violet')
plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')

plt.subplot(235)
plt.plot(x, normalization_y, 'g')
plt.title('双边振幅谱(归一化)', fontsize=9, color='green')

plt.subplot(236)
plt.plot(half_x, normalization_half_y, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
plt.savefig('1.pdf')
plt.show()

plt.figure()
f, t, nd = signal.stft(y, fs, nperseg=512, return_onesided=True)
#plt.pcolormesh(t, f, np.abs(nd), vmin = 0, vmax = 4, shading='gouraud')
plt.pcolormesh(t, f, np.abs(nd), shading='gouraud')
plt.title('STFT')
plt.ylabel('frequency (Hz)')
plt.xlabel('time (s)')
plt.savefig('2.pdf')
plt.show()


def complexsig_addnoise(s, snr):
    #实部加噪
    s_r = np.real(s)
    psr = np.sum(np.abs(s_r)**2)/ len(s_r)
    pnr = psr / (np.power(10, snr / 10))
    noise_r = np.random.randn(len(s_r)) * np.sqrt(pnr)

    #虚部加噪
    s_im = np.imag(s)
    psim = np.sum(np.abs(s_im)**2)/ len(s_im)
    pnim = psim / (np.power(10, snr / 10))
    noise_im = np.random.randn(len(s_im)) * np.sqrt(pnim)

    #构成复数噪声信号
    noise = noise_r +1j*noise_im

    sn = s + noise
    return sn
