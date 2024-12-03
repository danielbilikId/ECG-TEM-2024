clc; clear all; close all;

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

% ============================ Section 2: Fourier Coefficient Calculation =========================
% Define parameters for Fourier analysis
K = 5; % Number of harmonics to consider
Kmax = 4 * K + 2; % Maximum number of Fourier components
m = -Kmax:Kmax; % Frequency indices for Fourier series

% Compute Fourier coefficients (G)
G = zeros(1, length(m)); % Initialize Fourier coefficients
for n = 1:N
    G = G + x(n) * exp(-2 * pi * 1i * n .* m ./ N); % Discrete Fourier Transform
end

% Reconstruct signal (y) from Fourier coefficients
y = zeros(1, N); % Initialize reconstructed signal
for i = 1:length(m)
    y = y + G(i) .* exp(1i * m(i) * 2 * pi / T * t); % Inverse Fourier Transform
end
y = real(y) / N; % Take real part and normalize

% =============================== Section 3: TEM Sampling and Estimation ==========================
% Define TEM parameters
b = 2.9; % TEM bias
d = 0.08; % TEM threshold
kappa = 0.5; % TEM scaling factor

% Perform TEM sampling
[tnIdx, yInt] = iafTEM(y, dt, b, d, kappa); % Obtain firing times and integrated signal
tn = t(tnIdx); % Convert indices to time values
Ntn = length(tn); % Number of firing times

% Generate differential delay sequence (yDel)
yDel = -b * diff(tn) + kappa * d; % Differential delays

% Define Fourier sample recovery parameters
w0 = 2 * pi / T; % Fundamental frequency
F = exp(1j * w0 * tn(2:end)' * (-Kmax:Kmax)) - exp(1j * w0 * tn(1:end-1)' * (-Kmax:Kmax)); % Fourier basis
F(:, Kmax+1) = tn(2:end) - tn(1:end-1); % Add temporal differences
s = T ./ (1j * 2 * pi * (-Kmax:Kmax)); % Scale factor for Fourier coefficients
s(Kmax+1) = 1;
S = diag(s); % Diagonal scaling matrix

% Solve for Fourier coefficients (ytnHat)
ytnHat = pinv(F * S) * yDel'; % Solve system of equations for Fourier coefficients
ytnHat = ytnHat' * N; % Normalize coefficients

% Process spectrum for denoising and dimensionality reduction
spectrum = conj(ytnHat(Kmax+1:end)); % Extract positive frequency spectrum
spectrum = cadzow(spectrum, Kmax-2, inf)'; % Apply Cadzow denoising

% =============================== Section 4: Parameter Estimation ================================
% Construct Toeplitz matrix for parameter recovery
l = round(length(spectrum) / 2) * 2; % Adjust length
tr = flip(spectrum(1:(length(spectrum)) / 2 + 2)); % First half of spectrum (flipped)
tc = spectrum((length(spectrum)) / 2 + 2:end); % Second half of spectrum
tt = toeplitz(tc, tr); % Construct Toeplitz matrix

% Perform SVD on Toeplitz matrix
[U, S, V] = svd(tt); % Singular Value Decomposition
% Validate K against the number of columns in V
[~, ~, V] = svd(tt); % Perform SVD on Toeplitz matrix
num_singular_vectors = size(V, 2); % Number of singular vectors
if K > num_singular_vectors
    error('K exceeds the number of singular vectors in V. Reduce K or increase spectrum length.');
end

V = conj(V(:, 1:K))'; % Take first K singular vectors
V(1,:) = -V(1,:); % Adjust sign convention
m = length(tr); % Length of the spectrum
V = V'; % Transpose for computation
v1 = V(1:m-1,:); % Submatrix 1
v2 = V(2:m,:); % Submatrix 2

% Estimate roots and parameters
[v, w] = eig(pinv(v2) * v1); % Eigenvalue decomposition
w = conj(w); % Conjugate eigenvalues
ww = diag(w); % Extract eigenvalues
uk = ww'; % Eigenvalues (roots of annihilating filter)

uk = esprit(spectrum, K); % Alternative estimation using ESPRIT
tk = T * atan2(imag(uk), real(uk)) / (2 * pi); % Compute delays
rk = -T * log(abs(uk)) / (2 * pi); % Compute pulse widths

% Ensure valid pulse widths
for k = 1:K
    if rk(k) <= 0
        rk(k) = T / N; % Assign minimum pulse width
    end
end

% Estimate amplitudes (c_k)
ck = 1 / T * pinv(vander2(uk, length(spectrum)))' * spectrum';
tk = mod(tk, T); % Wrap delays to [0, T]
rk = sort(rk, 'descend'); % Sort pulse widths
ck = ck / N; % Normalize amplitudes
ck = abs(ck);
% =============================== Section 5: Signal Reconstruction ===============================
t_eval = linspace(0, T, N); % Evaluation time vector

% Optimization options
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
    'Display', 'iter', 'MaxIterations', 1000, 'OptimalityTolerance', 1e-6);

% Combine ck, tk, rk into a single vector for optimization
params_init = [ck, tk, rk];

% Define the objective function using pointwise error minimization
objective = @(params) pointwise_error(params, t_eval, x, K);

% Optimize parameters
best_params = fminunc(objective, params_init, options);

% Extract optimized ck, tk, rk
ck_best = best_params(1:K);
tk_best = best_params(K+1:2*K);
rk_best = best_params(2*K+1:end);

% Reconstruct the signal with optimized parameters
signal2 = zeros(size(t_eval));
for k = 1:K
    signal2 = signal2 + time_eval(ck_best, rk_best, tk_best, t_eval, k);
end
signal2 = signal2/max(signal2); % Normalize reconstructed signal

% ================================ Section 6: Visualization =======================================
% Plot original and reconstructed signals
figure;
plot(t, x, 'k', 'LineWidth', 1.5); hold on;
plot(t, signal2, '--r', 'LineWidth', 1.5);
legend('Original Signal', 'Reconstructed Signal');
xlabel('Time (s)');
ylabel('Amplitude');
title('Original vs. Reconstructed ECG Signal');
grid on;

% ============================

function error = pointwise_error(params, t_eval, x, K)
    % Objective function to minimize the pointwise error
    % Inputs:
    % params - Combined vector of ck, tk, rk
    % t_eval - Time vector for reconstruction
    % x      - Original signal
    % K      - Number of pulses
    % Output:
    % error - Sum of absolute differences between signals
    
    % Extract ck, tk, rk from params
    ck = params(1:K);
    tk = params(K+1:2*K);
    rk = params(2*K+1:end);
    
    % Reconstruct the signal
    signal2 = zeros(size(t_eval));
    for k = 1:K
        signal2 = signal2 + time_eval(ck, rk, tk, t_eval, k);
    end
    
    % Normalize reconstructed signal
    signal2 = signal2 / max(signal2);
    
    % Compute pointwise error as sum of absolute differences
    error = sum(abs(x' - signal2));
end