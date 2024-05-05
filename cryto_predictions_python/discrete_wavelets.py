import pywt
import pywt.data
import numpy as np
import ast
from ast import literal_eval
from matplotlib import pyplot as plt

# # String representation of the list of lists
# with open('test_data/test_array.txt', 'r') as f:
#     txt = f.read()
#
# # Safely evaluate the string into a Python list
# list_data = ast.literal_eval(txt)
# # Convert the Python list into a NumPy array
# original = np.array(list_data)
# print(original.shape)
#
# # Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# coeffs2 = pywt.dwt2(original,  'rbio2.6')
# LL, (LH, HL, HH) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# fig.tight_layout()
# plt.show()

array = [[9, 7, 6, 2], [5, 3, 4, 4], [8, 2, 4, 0], [6, 0, 2, 2], [3, 0, 25, 3]]

# cA, cD = pywt.dwt(array, 'db1', 'zero')
# print('length: {}',len(cA));
# print('db 1')
# print(cA)
# print(cD)
#
w = pywt.Wavelet('db2')
w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)

print(w.filter_bank)

#
print(pywt.families())
print(pywt.wavelist('gaus'))
print(pywt.wavelist('mexh'))
print(pywt.wavelist('cgau'))
print(pywt.wavelist('shan'))
print(pywt.wavelist('fbsp'))
print(pywt.wavelist('cmor'))

#
# # # for mode_name in ['zero', 'constant', 'symmetric', 'reflect', 'periodic', 'smooth', 'periodization']:
import matplotlib.pyplot as plt
# lb = -5
# ub = 5
# n = 10
# wavelet = pywt.ContinuousWavelet("gaus8")
# wavelet.upper_bound = ub
# wavelet.lower_bound = lb
# [psi,xval] = wavelet.wavefun(length=n)
#
# print('psi: {}', psi)
#
# print('xval: {}', xval)

# plt.plot(xval,psi) # doctest: +ELLIPSIS
# plt.title("Gaussian Wavelet of order 8")
# plt.show()

coeffs2 = pywt.dwt2(array,  'db2', 'zero')
LL, (LH, HL, HH) = coeffs2
print('LL coefs transform')
print(LL)
print('LH coefs transform')
print(LH)
print('HL coefs transform')
print(HL)
print('HH coefs transform')
print(HH)


original_array = pywt.idwt2(coeffs2, 'db2', 'zero')
print('inverse transform')
print(original_array)




