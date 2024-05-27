
import pywt
import numpy as np

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

# wavelist = pywt.wavelist(kind='discrete')
# print('wavelist', wavelist)

wavelet = pywt.Wavelet('db2')

[dec_lo, dec_hi, rec_lo, rec_hi] = wavelet.filter_bank

print('dec_lo', dec_lo)
print('dec_hi', dec_hi)
print('rec_lo', dec_lo)
print('rec_hi', dec_hi)


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

w_type = 'db3'
mode = 'reflect'
sig = np.array([[1.1515151515151515, 2.2626262626262626, 3.36363636363636363636, 4.0, 5.0, 6.0], 
                [6.616161616161616161, 7.7272727272727272, 8.818181818181818181, 9.0, 10.0, 11.0]])

wavelet = pywt.dwt2(sig, w_type, mode)
print('db2 wavelet', wavelet)

invesed = pywt.idwt2(wavelet, w_type, mode)
print('inversed', invesed)

sig = np.array([[6, 5, 4, 3]])
padded = pywt.pad(sig, (10, 6), 'reflect')
print('padded', padded)
