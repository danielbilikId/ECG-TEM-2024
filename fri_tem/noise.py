import numpy as np

def add_gaussian_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    """Adds Gaussian noise to the signal."""
    noise = noise_level * np.random.normal(0, 1, signal.shape)
    return signal + noise
