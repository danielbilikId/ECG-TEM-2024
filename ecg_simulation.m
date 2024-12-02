
% Main Script for VPW-FRI Signal Reconstruction

% Parameters
fs = 2000; % Sampling frequency (Hz)
T = 1 / fs; % Sampling period (s)
N = 512; % Number of samples
K = 5; % Number of pulses to reconstruct

% Generate synthetic VPW-FRI signal
t = (0:N-1) * T; % Time vector
true_t = [0.2, 0.5, 0.7, 1.0, 1.2]; % True pulse locations (seconds)
true_r = [0.02, 0.03, 0.015, 0.02, 0.025]; % True pulse widths
true_c = [1.5, -2.0, 1.0, -1.5, 2.5]; % True amplitudes
signal = zeros(1, N);

% Generate Lorentzian pulses
for k = 1:K
    signal = signal + (true_c(k) ./ (pi * (true_r(k)^2 + (t - true_t(k)).^2)));
end

% Simulate sampling by taking uniform samples
samples = signal + 0.01 * randn(1, N); % Add small noise

% Perform VPW-FRI reconstruction
[t_est, r_est, c_est] = VPW_FRI_Reconstruction(samples, N, T, K);

% Display results
disp('True Pulse Locations:');
disp(true_t);
disp('Estimated Pulse Locations:');
disp(t_est);

disp('True Pulse Widths:');
disp(true_r);
disp('Estimated Pulse Widths:');
disp(r_est);

disp('True Pulse Amplitudes:');
disp(true_c);
disp('Estimated Pulse Amplitudes:');
disp(c_est);

% Reconstruct signal from estimated parameters
reconstructed_signal = zeros(1, N);
for k = 1:K
    reconstructed_signal = reconstructed_signal + ...
        (real(c_est(k)) ./ (pi * (real(r_est(k))^2 + (t - real(t_est(k))).^2)));
end

% Plot original and reconstructed signals
figure;
plot(t, signal, 'b', 'LineWidth', 1.5); hold on;
plot(t, reconstructed_signal, 'r--', 'LineWidth', 1.5);
legend('Original Signal', 'Reconstructed Signal');
title('VPW-FRI Signal Reconstruction');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
