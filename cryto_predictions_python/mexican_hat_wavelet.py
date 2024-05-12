
import pywt
import numpy as np

widths = np.arange(1, 6)

print(widths)


sig = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
sig = np.array([1, 2, 3])

print('sig shape', sig.shape)
print('signal', sig)

cwtmatr, freqs = pywt.cwt(sig, widths, 'morl')

print('cwtmatr', cwtmatr)
print('freqs', freqs)

maxArg = np.argmax(sig);

print('maxArg', maxArg)

