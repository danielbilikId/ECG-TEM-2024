import numpy as np
import scipy.io
import scipy.signal

def estimate_period(ecg_signal, fs):
    """
    Estimates the period (pulse repetition interval) of ECG data.
    
    Parameters:
    - ecg_signal: array-like, the ECG data.
    - fs: int, the sampling rate of the ECG data in Hz.
    
    Returns:
    - T_estimated: float, the estimated period of ECG pulses in seconds.
    """
    ecg_signal = ecg_signal - np.mean(ecg_signal)  
    ecg_signal = ecg_signal / np.max(np.abs(ecg_signal))

    peaks, _ = scipy.signal.find_peaks(ecg_signal, distance=0.6*fs, height=0.5)
    rr_intervals = np.diff(peaks) / fs 

    T_estimated = np.mean(rr_intervals)
    return T_estimated