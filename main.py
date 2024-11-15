import numpy as np
from fri_tem.ecg_loader import load_ecg_signal
from fri_tem.estimate_signal import estimate_signal
from fri_tem.spectrum import sepctrum
from fri_tem.plotter import plot_reconstruction
import matplotlib.pyplot as plt
np.random.seed(32)

def main():
    K = 7
    T = 1
    b = 1.0
    d = 0.9
    kappa = 0.018
    noise_level = 0.1

    signal = load_ecg_signal('u.mat', 'signal2',noise_level)
    tn, spectrum_estimated = sepctrum(signal,K,Kmax=4*K+2,b=b,d=d,kappa=kappa,T=T)
    signal_estimated = estimate_signal(signal,spectrum_estimated,T,K)
    plot_reconstruction(signal, signal_estimated, tn, output_path="output/reconstructed_data.mat",T=T)

if __name__ == "__main__":
    main()

