clear
clc
close all

% Step 1: Load and Filter ECG Signal
fs = 2000; % Sampling frequency
t = 0:1/fs:1; % Time vector
T_period = 1; % Signal period
x = load('u.mat'); % Load ECG signal
x = real(x.signal2);
K = 10; % Number of ECG pulses to reconstruct

% Step 2: Band-Pass Filter
f_low = 0.5; f_high = 40; % Band-pass filter frequencies
[b, a] = butter(4, [f_low, f_high] / (fs / 2), 'bandpass');
x_filtered = filtfilt(b, a, x);

% Step 3: Simulate TEM Sampling
b_t = 1.24; % Bias
delta = 0.05; % Threshold
kappa = 1; % Scaling factor
integrator = 0;
t_n = []; % Time instances of firing
for i = 2:length(x_filtered)
    integrator = integrator + kappa * (x_filtered(i) + b_t) / fs;
    if integrator >= delta
        t_n = [t_n, t(i)];
        integrator = 0; % Reset integrator
    end
end

% Step 4: Compute Fourier Coefficients from TEM Samples
M = 2 * K; % Number of Fourier coefficients
omega_0 = 2 * pi / T_period; % Fundamental frequency
X = zeros(1, M);
for m = 1:M
    X(m) = sum(exp(-1j * (m - 1) * omega_0 * t_n)) / T_period;
end

% Step 5: Apply IQML Denoising
X_denoised = iqml_denoising(X, K);

% Step 6: Recover VPW-FRI Parameters
params_recovered = vpw_fri_reconstruction(X_denoised, M, T_period, K, omega_0);

% Step 7: Define ECG Pulse Model for Reconstruction
ecg_pulse_model = @(t, c, d, r, T) ...
    (c .* r) ./ (pi * (r.^2 + (t - T).^2)) + ...
    (d .* (t - T)) ./ (pi * (r.^2 + (t - T).^2));

% Step 8: Reconstruct Signal Using Optimized Parameters
c_optimized = params_recovered.c;
d_optimized = params_recovered.d;
r_optimized = params_recovered.r;
T_k_optimized = params_recovered.T;

x_reconstructed = zeros(size(t));
for k = 1:K
    x_reconstructed = x_reconstructed + ...
        (c_optimized(k) * r_optimized(k)) ./ (pi * (r_optimized(k)^2 + (t - T_k_optimized(k)).^2)) + ...
        (d_optimized(k) * (t - T_k_optimized(k))) ./ (pi * (r_optimized(k)^2 + (t - T_k_optimized(k)).^2));
end

% Step 9: Normalize Reconstructed Signal
scaling_factor = max(abs(x)) / max(abs(x_reconstructed));
x_reconstructed = x_reconstructed * scaling_factor;

% Step 10: Plot Results
figure;
subplot(3, 1, 1); plot(t, x, 'LineWidth', 1.5); title('Original ECG Signal');
subplot(3, 1, 2); plot(t, x_filtered, 'LineWidth', 1.5); title('Filtered ECG Signal');
subplot(3, 1, 3); plot(t, x_reconstructed, 'LineWidth', 1.5); title('Reconstructed ECG Signal');

% VPW-FRI Reconstruction Function
function recovered_params = vpw_fri_reconstruction(X, M, T_period, K, omega_0)
    % Construct Toeplitz Matrix for Annihilating Filter
    X_toeplitz = hankel(X(1:K), X(K:M)); % Hankel matrix for robustness
    [~, ~, V] = svd(X_toeplitz); % Singular value decomposition
    annihilating_filter = V(:, end); % Last column is null space

    % Find Roots of Annihilating Filter
    roots_filter = roots(annihilating_filter);

    % Filter Spurious Roots
    valid_roots = roots_filter(abs(imag(roots_filter)) > 1e-3); % Avoid real roots
    [~, idx] = sort(abs(valid_roots), 'descend');
    roots_filter = valid_roots(idx(1:K));

    % Compute Delays and Widths
    T_k = mod(-angle(roots_filter) * T_period / (2 * pi), T_period); % Delays within [0, T_period]
    r_k = -log(abs(roots_filter)) * T_period / (2 * pi); % Pulse widths

    % Recover Amplitudes (c_k and d_k)
    A_matrix = zeros(M, K);
    for m = 1:M
        for k = 1:K
            A_matrix(m, k) = exp(-2 * pi * (r_k(k) * abs(m - 1) + 1j * T_k(k) * (m - 1)) / T_period);
        end
    end
    c_d = pinv(A_matrix) * X(1:M).'; % Solve linear system
    c_k = real(c_d); % Symmetric amplitudes
    d_k = imag(c_d); % Asymmetric amplitudes

    % Output recovered parameters
    recovered_params = struct('T', T_k, 'r', r_k, 'c', c_k, 'd', d_k);
end
function X_denoised = iqml_denoising(X, K)
    % IQML denoising based on iterative quadratic maximum likelihood
    max_iters = 10; % Maximum number of iterations
    tol = 1e-6; % Convergence tolerance
    M = length(X); % Length of Fourier coefficients

    % Ensure Fourier coefficients are valid
    if any(isnan(X)) || any(isinf(X))
        error('Fourier coefficients (X) contain NaN or Inf.');
    end

    % Step 1: Construct initial Hankel matrix
    try
        S = hankel(X(1:K), X(K:M));
    catch
        error('Error constructing Hankel matrix. Check dimensions of X and K.');
    end

    % Iterative process
    for iter = 1:max_iters
        % Enforce rank-K constraint using SVD
        try
            [U, S_mat, V] = svd(S, 'econ');
        catch
            error('SVD failed. Check the Hankel matrix (S) for ill-conditioning.');
        end

        S_denoised = U(:, 1:K) * S_mat(1:K, 1:K) * V(:, 1:K)';

        % Enforce Hankel structure by averaging along anti-diagonals
        X_denoised = zeros(1, M);
        for diag_idx = 1:(size(S_denoised, 1) + size(S_denoised, 2) - 1)
            [rows, cols] = find(bsxfun(@eq, (1:size(S_denoised, 1))' + (1:size(S_denoised, 2)), diag_idx));
            values = S_denoised(sub2ind(size(S_denoised), rows, cols));
            values = values(~isnan(values) & ~isinf(values)); % Filter valid values
            if ~isempty(values)
                X_denoised(diag_idx) = mean(values);
            else
                X_denoised(diag_idx) = 0; % Set to zero if no valid values
            end
        end

        % Update the Hankel matrix with the new denoised coefficients
        try
            S = hankel(X_denoised(1:K), X_denoised(K:M));
        catch
            error('Error updating Hankel matrix. Check dimensions of X_denoised.');
        end

        % Check for convergence
        if norm(S - S_denoised, 'fro') / norm(S, 'fro') < tol
            break;
        end

        % Validate after iteration
        if any(isnan(S(:))) || any(isinf(S(:)))
            error('Hankel matrix (S) contains NaN or Inf during iteration.');
        end
    end

    % Final validation
    if any(isnan(X_denoised)) || any(isinf(X_denoised))
        error('Final denoised Fourier coefficients contain NaN or Inf.');
    end
end

