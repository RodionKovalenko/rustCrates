
import pywt
import numpy as np

widths = np.arange(1, 6)



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

sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8, 9]])
sig = np.array([1, 2, 3])


print('data', sig)
print('scales', scales)
wavelet, frequencies = pywt.cwt(sig, scales, 'cgau8')
print('wavelet', wavelet)
print('frequencies', frequencies)
complex = 0.07660532 -0.12197029j;

print('complex',complex.real)
print('complex',complex.imag)
#wavelet [[ 0.07660532+0.12197029j  0.04833911+0.13560878j -0.36331012+0.22956201j]
 #[-0.02201995-0.14426308j -0.40243903+0.14258296j  0.5893798 +0.44028149j]]
# CGAU8 [[Complex { re: 0.0766053184239858, im: -0.12197028564451548 }, Complex { re: 0.048339111819366315, im: -0.13560877652487657 }, 
# Complex { re: -0.3633101221491256, im: -0.2295620132786386 }], [Complex { re: -0.02201995070211324, im: 0.14426308336102447 }, 
#                                                                 Complex { re: -0.40243903345004584, im: -0.14258296031136422 },
#                                                                   Complex { re: 0.5893797983054468, im: -0.4402814940342556 }]]
# #[0.7, 0.35]
