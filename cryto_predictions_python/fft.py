import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 8  # Sampling rate in Hz
T = 1.0 / sampling_rate  # Sampling interval
t = np.arange(0.0, 1.0, T)  # Time vector (1 second duration)

# Signal generation: 50 Hz and 120 Hz components
freq1 = 50  # Frequency in Hz
freq2 = 120  # Frequency in Hz
signal = 0.6 * np.sin(2 * np.pi * freq1 * t) + 0.9 * np.sin(2 * np.pi * freq2 * t)

signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
print('signal', signal)

# Compute the Fast Fourier Transform (FFT)
fft_values = np.fft.fft(signal)
n = len(signal)
frequencies = np.fft.fftfreq(n, T)  # Frequency axis

# Only take the positive half of the spectrum, since it's symmetric for real signals
half_n = n // 2  
fft_values = fft_values[:half_n]
frequencies = frequencies[:half_n]

print('fft values', fft_values)

# # Plotting the results
# plt.figure(figsize=(12, 6))

# # Time-domain signal
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Time Domain Signal')
# plt.xlabel('Time [seconds]')
# plt.ylabel('Amplitude')

# # Frequency-domain signal
# plt.subplot(2, 1, 2)
# plt.stem(frequencies, np.abs(fft_values), 'b', markerfmt=" ", basefmt="-b")
# plt.title('Frequency Spectrum')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude')
# plt.tight_layout()

# plt.show()
