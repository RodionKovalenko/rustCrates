
import pywt
import numpy as np

widths = np.arange(1, 6)
sig = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print('widths', widths)
print('sig shape', sig.shape)
print('signal', sig)

wavelist = pywt.wavelist(kind='continuous')
print('wavelist', wavelist)

cwtmatr, freqs = pywt.cwt(sig, widths, 'gaus8')

print('cwtmatr', cwtmatr)
print('freqs', freqs)

print(np.e)
print(np.pi)

wavelet = pywt.ContinuousWavelet('gaus1')

print(wavelet.wavefun(10))

data = np.random.rand(2, 3, 1, 3, 4)
data = np.array([[[[[0.64628843, 0.82996726, 0.87006796, 0.48253978],
                      [0.43919335, 0.06566719, 0.06158033, 0.81224427],
                      [0.88918919, 0.02075738, 0.58363584, 0.37932007]]],

                    [[[0.19943944, 0.37560552, 0.09522572, 0.62038108],
                      [0.6841884 , 0.01350215, 0.27384274, 0.69728249],
                      [0.53189142, 0.99672615, 0.03854367, 0.30125334]]],

                    [[[0.68961004, 0.59868674, 0.00769092, 0.51516282],
                      [0.8776632 , 0.70469546, 0.40765667, 0.37929271],
                      [0.43646761, 0.13832165, 0.15463361, 0.37499216]]]],


                   [[[[0.41296786, 0.42958691, 0.1443393 , 0.9200188 ],
                      [0.03331772, 0.86528934, 0.18012958, 0.35431852],
                      [0.7720741 , 0.410375  , 0.37368207, 0.82489805]]],

                    [[[0.13901775, 0.80671067, 0.04599183, 0.5085041 ],
                      [0.69940342, 0.68145641, 0.73054897, 0.34058879],
                      [0.19212236, 0.19808242, 0.38803242, 0.66113789]]],

                    [[[0.06598777, 0.37103162, 0.06875532, 0.06488421],
                      [0.71048144, 0.97461436, 0.14561172, 0.56559529],
                      [0.02740725, 0.95295214, 0.76303086, 0.96808568]]]]])

print("Shape of the 5D array:", data.shape)
print("Example of the 5D array:", data)

# Convert the data to a numpy array
numpy_array = np.array(data)
cwtmatr, freqs = pywt.cwt(data, widths, 'gaus8')

print('result', cwtmatr)
