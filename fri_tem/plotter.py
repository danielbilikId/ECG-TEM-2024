import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.io as sio

def plot_reconstruction(signal, reconstructed_signal, tn, output_path, T):
    """
    Plots and saves the reconstruction results, scaling the reconstructed signal
    to match the min and max values of the original signal.
    
    Parameters:
    - signal: np.ndarray, the original ECG signal.
    - reconstructed_signal: np.ndarray, the reconstructed ECG signal.
    - tn: np.ndarray, sample times.
    - output_path: str, path to save the plot and data.
    - T: float, total time duration of the signal.
    """
    # Align the min and max points of the reconstructed signal to the original signal
    signal_min, signal_max = np.min(signal), np.max(signal)
    reconstructed_min, reconstructed_max = np.min(reconstructed_signal), np.max(reconstructed_signal)
    
    # Scale the reconstructed signal to match the original signal's amplitude range
    reconstructed_signal_scaled = ((reconstructed_signal - reconstructed_min) / 
                                   (reconstructed_max - reconstructed_min)) * (signal_max - signal_min) + signal_min
    
    # Centering the baseline by aligning the mean values
    reconstructed_signal_scaled -= np.mean(reconstructed_signal_scaled) - np.mean(signal)
    
    # Time axis
    N = len(signal)
    time = np.linspace(0, T, N)
    
    # Create subplots
    plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2)
    
    # Plot the original signal
    classical_vpw = plt.subplot(gs[0, 0])
    classical_vpw.plot(time, signal, label="Original Signal", color="blue")
    classical_vpw.set_title("Original Signal")
    classical_vpw.set_xlabel("Time (s)")
    classical_vpw.set_ylabel("Amplitude")
    
    # Plot the reconstructed signal with scaling and baseline adjustment
    recovered_vpw = plt.subplot(gs[0, 1])
    recovered_vpw.plot(time, signal/np.max(signal), label="Original Signal", color="blue", alpha=0.5)
    recovered_vpw.plot(time, reconstructed_signal_scaled/np.max(reconstructed_signal_scaled), label="Reconstructed Signal", color="orange", alpha=0.75)
    recovered_vpw.set_title("Reconstructed Signal (Aligned to Original)")
    recovered_vpw.set_xlabel("Time (s)")
    recovered_vpw.set_ylabel("Amplitude")
    
    # Add legends and grid
    classical_vpw.legend()
    recovered_vpw.legend()
    classical_vpw.grid(True)
    recovered_vpw.grid(True)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    