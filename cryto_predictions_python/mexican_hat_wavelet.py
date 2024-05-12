
import pywt
import numpy as np

widths = np.arange(5, 10)
sig = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
sig = np.array([[123, 253, 523], [134, 88, 46]])

print('widths', widths)
print('sig shape', sig.shape)
print('signal', sig)

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

cwtmatr, freqs = pywt.cwt(sig, widths, 'cmor')

print('cwtmatr', cwtmatr)
print('freqs', freqs)