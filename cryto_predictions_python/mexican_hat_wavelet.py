
import pywt
import numpy as np

widths = np.arange(1, 6)



wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

wavelist = pywt.wavelist(kind='discrete')
print('wavelist', wavelist)


mode = 'constant'
sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8, 9])
print('data', sig)
ca, cd = pywt.dwt(sig, 'db3', mode)
print('ca', ca)
print('cd', cd)


sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8, 9]])
print('data', sig)
ca, cd = pywt.dwt2(sig, 'db5', mode)
print('ca', ca)
print('cd', cd)