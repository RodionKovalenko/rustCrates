import pywt

array = [1, 2, 3, 4, 5, 6]
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# cA, cD = pywt.dwt(array, 'db1', 'zero')
# print('length: {}',len(cA));
# print('db 1')
# print(cA)
# print(cD)
#
w = pywt.Wavelet('bior2.6')
w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)

print(w.filter_bank)

#
# print(pywt.families())
# print(pywt.wavelist('bior'))
#
# # # for mode_name in ['zero', 'constant', 'symmetric', 'reflect', 'periodic', 'smooth', 'periodization']:
# cA2, cD2 = pywt.dwt(array, 'bior2.6')
#
# print('db2')
# print(cA2)
# print(cD2)
#
# original_array = pywt.idwt(cA2, cD2, 'bior2.6')
#
# print('db2 inverse')
# print(original_array)

