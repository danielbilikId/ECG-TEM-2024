import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_reconstruction(signal, reconstructed_signal, tn, output_path,T):
    """Plots and saves the reconstruction results."""
    N = len(signal)
    gs = gridspec.GridSpec(1, 2)
    classical_vpw = plt.subplot(gs[0, 0])
    time = np.linspace(0, T, N)
    classical_vpw.plot(time,signal/np.max(signal))
    recovered_vpw = plt.subplot(gs[0, 1])
    recovered_vpw.plot(time,(signal)/np.max(signal))
    recovered_vpw.plot(time,reconstructed_signal)
    plt.show()
