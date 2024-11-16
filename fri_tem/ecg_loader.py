import numpy as np
import scipy.io
from scipy.signal import resample, detrend, savgol_filter
from fri_tem.noise import add_gaussian_noise
def load_ecg_signal(file_path: str, signal_key: str, noise_level: float ,resample_size: int = 200) -> np.ndarray:
    """Loads and preprocesses the ECG signal from a .mat file."""
    data = scipy.io.loadmat(file_path)
    signal = data[signal_key].flatten()
    signal[np.isnan(signal)] = 0
    signal = add_gaussian_noise(signal,noise_level)
    return resample(signal, resample_size)


def process_ecg_signal(signal: np.ndarray, resample_size: int = 200) -> np.ndarray:
    """
    Processes the ECG signal by removing NaNs, detrending, centering, and smoothing.
    
    Parameters:
    - signal: np.ndarray, the raw ECG signal.
    - resample_size: int, the number of samples to resample the signal to.
    
    Returns:
    - np.ndarray, the processed ECG signal.
    """
    signal[np.isnan(signal)] = 0
    
    signal = resample(signal, resample_size)
    
    signal = detrend(signal)
    
    signal = signal - np.mean(signal)
    
    signal = savgol_filter(signal, window_length=11, polyorder=2)
    
    return signal
