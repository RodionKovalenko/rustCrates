
import pywt
import numpy as np
widths = np.arange(1, 6)

print(widths)


sig = np.arange(1, 8)
print('signal')
print(sig)

cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')

print('cwtmatr')
print(cwtmatr)
print('freqs')
print(freqs)
