import pywt

array = [1, 2, 3, 4, 5, 6]
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# cA, cD = pywt.dwt(array, 'db1', 'zero')
# print('length: {}',len(cA));
# print('db 1')
# print(cA)
# print(cD)
#
w = pywt.Wavelet('db8')
w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)

print(w.filter_bank)

#
print(pywt.families())
print(pywt.wavelist('gaus'))
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



cA2, cD2 = pywt.dwt(array, 'db8', 'periodic')

print('db2')
print(cA2)
print(cD2)

original_array = pywt.idwt(cA2, cD2, 'db8', 'periodic')

print('db2 inverse')
print(original_array)

w = pywt.Wavelet('db2')
print(w.orthogonal)

(phi, psi, x) = w.wavefun(level=2)

print('phi')
print(phi)

print('psi')
print(psi)

print('x')
print(x)