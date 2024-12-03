clc; clear all; close all;
rng(11)
% ================================ Section 1: Signal Preprocessing ================================
% Load ECG signal
signal = load("C:\Users\danie\workspace\ecg\GDN0001\GDN0001_2_Valsalva.mat"); 
x = signal.tfm_ecg2(1:1600); % Extract signal as column vector
x = x / max(x); % Normalize the signal to [0, 1]

% Generate time vector
N = length(x); % Number of samples
T = 1; % Signal period
dt = 1 / N; % Time step
t = linspace(0, T, N); % Time vector (from 0 to T)

% ============================ Section 2: Kernel Design ===========================
% Original Gaussian Kernel
kernel_width = 0.02; % Kernel width
kernel = exp(-((t - mean(t)).^2) / (2 * kernel_width^2)); % Gaussian kernel
kernel = kernel / sum(kernel); % Normalize kernel
x_original = conv(x, kernel, 'same'); % Convolve signal with kernel
x_original = x_original / max(x_original); % Normalize filtered signal

% Weighted Kernel (based on SNR)
freqs = linspace(0, N/2, N/2 + 1); % Frequency range
SNR = 10 + 1 * sin(2 * pi * freqs / max(freqs)); % Hypothetical SNR profile
SNR_weight = SNR / max(SNR); % Normalize SNR weights
fft_kernel = fft(kernel); % FFT of Gaussian kernel
weighted_fft_kernel = fft_kernel(1:N/2 + 1) .* SNR_weight; % Apply SNR weighting
weighted_kernel = ifft([weighted_fft_kernel, conj(weighted_fft_kernel(end-1:-1:2))]); % Inverse FFT
weighted_kernel = real(weighted_kernel); % Remove imaginary part
weighted_kernel = weighted_kernel / sum(weighted_kernel); % Normalize
x_weighted = conv(x, weighted_kernel, 'same'); % Convolve signal with weighted kernel
x_weighted = x_weighted / max(x_weighted); % Normalize weighted signal

% ================================ Section 3: Reconstruction at Different SNR Levels ==============
snr_levels = [5, 15, 30]; % SNR levels in dB
reconstructed_signals = cell(3, length(snr_levels));
original_signals = cell(3,length(snr_levels));
K = 7; 
for snr_idx = 1:length(snr_levels)
    % Add noise to original signal
    noise_power = 10^(-snr_levels(snr_idx) / 10);
    noisy_signal = x + sqrt(noise_power) * randn(size(x));
    original_signals{1,snr_idx} = noisy_signal;

    % No Kernel Reconstruction
    reconstructed_signals{1, snr_idx} = reconstruct_signal(noisy_signal', t, N, T,K);
    
    % Original Kernel Reconstruction
    noisy_signal_filtered = conv(noisy_signal, kernel, 'same');
    noisy_signal_filtered = noisy_signal_filtered / max(noisy_signal_filtered); % Normalize
    original_signals{2,snr_idx} = noisy_signal_filtered;
    reconstructed_signals{2, snr_idx} = reconstruct_signal(noisy_signal_filtered', t, N, T,K);
    
    % Weighted Kernel Reconstruction
    noisy_signal_weighted = conv(noisy_signal, weighted_kernel, 'same');
    noisy_signal_weighted = noisy_signal_weighted / max(noisy_signal_weighted); % Normalize
    original_signals{3,snr_idx} = noisy_signal_weighted;
    reconstructed_signals{3, snr_idx} = reconstruct_signal(noisy_signal_weighted', t, N, T,K);
end

% ================================ Section 4: Visualization =======================================
figure;
kernels = {'No Kernel', 'Gaussian Kernel', 'Weighted Kernel'}; % Cell array of kernel names
for snr_idx = 1:length(snr_levels)
    for kernel_idx = 1:3
        subplot(length(snr_levels), 3, (snr_idx - 1) * 3 + kernel_idx);
        plot(t, original_signals{kernel_idx,snr_idx},'k', 'LineWidth', 2); hold on;
        plot(t, reconstructed_signals{kernel_idx, snr_idx}, '--r', 'LineWidth', 2);
        legend('Original Signal', 'Reconstructed Signal');
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('SNR=%ddB, Kernel=%s', snr_levels(snr_idx), kernels{kernel_idx}));
        grid on;
    end
end

% ============================ Section 5: Frequency Response of Kernels ===========================
figure;
freq_axis = linspace(0, N/2, N/2 + 1);
plot(freq_axis, abs(fft_kernel(1:N/2+1)), 'b', 'LineWidth', 1.5); hold on;
plot(freq_axis, abs(weighted_fft_kernel), 'r', 'LineWidth', 1.5);
legend('Original Kernel', 'SNR-Weighted Kernel');
title('Frequency Response of Kernels');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% ============================ Helper Function for Reconstruction =================================
