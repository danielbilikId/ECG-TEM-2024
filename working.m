% Enhanced ECG Signal Reconstruction Code
clc; clear all; close all;

% ================================ Section 1: Signal Preprocessing ================================
% Load ECG signal
signal = load('u.mat'); 
x = signal.signal2; % Extract signal segment
x = x';
x = x / max(x); % Normalize the signal
x = x + 0.1; % Add small bias to avoid negative values

% Generate time vector
N = length(x);

dt = 1/N; 
t = 0:1/N:1-1/N;

% ============================ Section 2: Fourier Coefficient Calculation =========================
% Parameters for Fourier analysis
K = 6; % Number of harmonics
T = 1; % Signal period
Kmax = 4 * K + 2; % Maximum number of Fourier components
m = -Kmax:Kmax; % Frequency indices
G = zeros(1, length(m)); % Initialize Fourier coefficients
Kmax = 3 * K + 2;

for n = 1:N
    G = G + x(n) * exp(-2 * pi * 1i * n .* m ./ N); 
end

y = 0;
for i = 1:1:length(m)
    y = y + G(i) .* exp(1i * m(i) * 2 * pi / T * t); 
end

y = real(y) / N;
b = 1.25; 
d = 0.08; 
kappa = 5e-1;

[tnIdx, yInt] = iafTEM(y, dt, b, d, kappa);
tn = t(tnIdx); 
Ntn = length(tn);
yDel = -b * diff(tn) + kappa * d;
K = Kmax; 
w0 = 2 * pi / T;

F = exp(1j * w0 * tn(2:end)' * (-K:K)) - exp(1j * w0 * tn(1:end-1)' * (-K:K));
F(:, K+1) = tn(2:end) - tn(1:end-1);
s = T ./ (1j * 2 * pi * (-K:K)); 
s(K+1) = 1;
S = diag(s);

ytnHat = pinv(F * S) * yDel';
ytnHat = ytnHat' * N;

spectrum = conj(ytnHat(K+1:end));
spectrum = cadzow(spectrum, K-2, inf)';
K = 11; 

swce = T * eye(length(ytnHat)) \ ytnHat';
swce = swce'; 

l = round(length(spectrum) / 2) * 2; 
tr = flip(spectrum(1:(length(spectrum)) / 2 + 2));
tc = spectrum((length(spectrum)) / 2 + 2:end);
tt = toeplitz(tc, tr); 
[U, S, V] = svd(tt);
V = conj(V(:, 1:K))';
V(1,:) = -V(1,:); 
m = length(tr); 
V = V'; 
v1 = V(1:m-1,:); 
v2 = V(2:m,:); 
[v, w] = eig(pinv(v2) * v1);
w = conj(w);
ww = diag(w);
uk = ww'; 

uk = esprit(spectrum, K);
tk = T * atan2(imag(uk), real(uk)) / (2 * pi);
rk = -T * log(abs(uk)) / (2 * pi);

for k = 1:K
    if rk(k) <= 0 
        rk(k) = T / N; 
    end
end

ck = 1 / T * pinv(vander2(uk, length(spectrum)))' * spectrum';
tk = mod(tk, T); 
tk2 = fliplr(tk); 
rk2 = fliplr(rk);
rk = fliplr(sort(rk2)); 
t = 0:1/N:1-1/N;
rk = rk + 0.0085;
ck = ck / N;


%% Step 4: Signal Reconstruction
% Permutations for parameter optimization
numPermutations = 20;
errorMin = inf;
bestSignal = [];
bestParams = [];

for permIdx = 1:numPermutations
    tk_perm = tk + 0.01 * (rand(1, K) - 0.5);
    rk_perm = rk + 0.01 * (rand(1, K) - 0.5);
    ck_perm = ck + 0.01 * (rand(1, K) - 0.5);
    
    % Reconstruct the signal
    t_eval = linspace(0, T, N);
    signal2 = zeros(size(t_eval));
    for k = 1:K
        signal2 = signal2 + time_eval(ck_perm, rk_perm, tk_perm, t_eval, k);
    end

    % Calculate reconstruction error
    error = norm(x - signal2);
    if error < errorMin
        errorMin = error;
        bestSignal = signal2;
        bestParams = [tk_perm; rk_perm; ck_perm];
    end
end

%% Step 5: Plot Results
figure;
plot(t, x, 'k', 'LineWidth', 1.5); hold on;
plot(t, bestSignal, '--r', 'LineWidth', 1.5);
legend('Original Signal', 'Reconstructed Signal');
xlabel('Time (s)');
ylabel('Amplitude');
title('Original vs. Reconstructed ECG Signal');
grid on;
