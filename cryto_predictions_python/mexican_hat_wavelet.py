
import pywt
import numpy as np

widths = np.arange(5, 100)
sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print('widths', widths)
print('sig shape', sig.shape)
print('signal', sig)

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

cwtmatr, freqs = pywt.cwt(sig, widths, 'gaus8')

print('cwtmatr', cwtmatr)
print('freqs', freqs)

print(np.e)
print(np.pi)

wavelet = pywt.ContinuousWavelet('gaus1')

print(wavelet.wavefun(10))

