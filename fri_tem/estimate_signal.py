import numpy as np
from misc.fri import FRI_estimator
from scipy.signal import resample
def estimate_signal(signal,spectrum,T,K,resample_rate=200):
    N = len(signal)
    frequencies = np.fft.fftfreq(int(N), T / N)
    omega = 2 * np.pi * frequencies / T
    if(resample_rate is not 0): signal = resample(signal,resample_rate)
    fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters_iqml(signal, spectrum)
    fri_estimated = fri_estimated[0]
    spectrum_estimated = fri_estimated.evaluate_Fourier_domain(omega)
    signal_estimated = np.real(np.fft.ifft(spectrum_estimated))
    return signal_estimated