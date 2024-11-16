from simulations_misc.ecg_dataloader import ecg_dataloader, select_pulse
from simulations_misc.config import ECG_PATH
import numpy as np
from fri_tem.ecg_loader import process_ecg_signal
from fri_tem.estimate_signal import estimate_signal
from fri_tem.spectrum import sepctrum
from fri_tem.plotter import plot_reconstruction
from simulations_misc.estimate_period import estimate_period
import matplotlib.pyplot as plt
np.random.seed(32)

def main():
    K = 7
    kappa = 0.018
    b = 1.25
    d = 0.9
    fs = 2000 #sample rate [Hz]

    data = ecg_dataloader(ECG_PATH)

    for subject, conditions in data.items():
        for condition, ecg_signal in conditions.items():
            try:
                T = estimate_period(ecg_signal,fs)
                if T  > 2.25:
                    T = 1 
                print(f"Estimate period: {T}")
                signal = select_pulse(ecg_signal,fs*T,fs)
                signal = process_ecg_signal(signal,resample_size=2000)
                tn, spectrum_estimated = sepctrum(signal,K,Kmax=4*K+2,b=b,d=d,kappa=kappa,T=T)
                signal_estimated = estimate_signal(signal,spectrum_estimated,T,K)
                output_path = f"output/{subject}_{condition}_reconstructed_data.mat"
                plot_reconstruction(signal, signal_estimated, tn, output_path,T)
                print(f"Reconstruction completed and saved for {subject} - {condition}")
                
            except Exception as e:
                print(f"Failed to process {subject} - {condition}: {e}")

if __name__ == "__main__":
    main()
