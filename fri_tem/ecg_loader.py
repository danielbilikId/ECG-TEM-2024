import numpy as np
import scipy.io
from scipy.signal import resample
from fri_tem.noise import add_gaussian_noise

def load_ecg_signal(file_path: str, signal_key: str, noise_level: float ,resample_size: int = 200) -> np.ndarray:
    """Loads and preprocesses the ECG signal from a .mat file."""
    data = scipy.io.loadmat(file_path)
    signal = data[signal_key].flatten()
    signal[np.isnan(signal)] = 0
    signal = add_gaussian_noise(signal,noise_level)
    return resample(signal, resample_size)
