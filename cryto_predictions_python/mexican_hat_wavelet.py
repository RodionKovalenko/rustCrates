
import pywt
import numpy as np

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

# wavelist = pywt.wavelist(kind='discrete')
# print('wavelist', wavelist)


mode = 'constant'
scales = np.arange(1, 3)

# sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8, 9])
# print('data', sig)
# ca, cd = pywt.dwt(sig, 'db3', mode)
# print('ca', ca)
# print('cd', cd)


# sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8, 9]])
# print('data', sig)
# ca, cd = pywt.dwt2(sig, 'db1', mode)
# print('ca', ca)
# print('cd', cd)
import cmath

sig = np.array([[1, 2, 3], [4, 5, 6]])


wavelet = pywt.dwt2(sig, 'db2', 'constant')
print('db2 wavelet', wavelet)
