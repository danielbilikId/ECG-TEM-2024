import numpy as np
from misc.fri import FRI_estimator

def estimate_signal(signal,spectrum,T,K):
    N = len(signal)
    frequencies = np.fft.fftfreq(int(N), T / N)
    omega = 2 * np.pi * frequencies / T

    fri_estimated = FRI_estimator(K, T, T / N, T / N).estimate_parameters_iqml2(signal, spectrum)
    fri_estimated = fri_estimated[0]
    spectrum_estimated = fri_estimated.evaluate_Fourier_domain(omega)
    signal_estimated = np.real(np.fft.ifft(spectrum_estimated))
    return signal_estimated