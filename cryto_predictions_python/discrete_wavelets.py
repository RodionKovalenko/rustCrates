import pywt

array = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]
# array = [1, 2, 3, 4, 5, 6]
cA, cD = pywt.dwt(array, 'db1', 'zero')
print('length: {}',len(cA));
print('db 1')
print(cA)
print(cD)

w = pywt.Wavelet('db2')
w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
print('db 2 filter: ')
print(w.filter_bank)

# for mode_name in ['zero', 'constant', 'symmetric', 'reflect', 'periodic', 'smooth', 'periodization']:
cA2, cD2 = pywt.dwt(array, 'db2', 'zero')

print('db2')
print(cA2)
print(cD2)


cA, cD = pywt.dwt(array, 'db4', 'zero')
print('db4')
print('length: {}',len(cA));
print(cA)
print(cD)

cA, cD = pywt.dwt(array, 'db8', 'zero')

print('db8')
print('length: {}',len(cA));
print(cA)
print(cD)

original_array = pywt.idwt(cA2, cD2, 'db2', 'zero')
print('inverse transformed  of db2')
print(original_array)


print('db16')
cA, cD = pywt.dwt(array, 'db16', 'zero')
print('length: {}',len(cA));
print(cA)
print(cD)

original_array = pywt.idwt(cA, cD, 'db16', 'zero')
print('inverse transformed  of db16')
print(original_array)

print('db25')
cA, cD = pywt.dwt(array, 'db25', 'zero')
print('length: {}',len(cA));
print(cA)
print(cD)

print('bior 1_3')
cA, cD = pywt.dwt(array, 'bior1.3', 'zero')
print('length: {}',len(cA));
print(cA)
print(cD)

original_array = pywt.idwt(cA, cD, 'bior1.3', 'zero')
print('inverse transformed  of bior1.3')
print(original_array)

w = pywt.Wavelet('bior1.3')
w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
print('bior1.3 filter: ')
print(w.filter_bank)