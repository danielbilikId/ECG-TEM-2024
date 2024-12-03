function [reconstructed_signal, tk_best, rk_best, ck_best] = reconstruct_signal(x, t, N, T, K)
    % Function to reconstruct a signal using Fourier coefficients and optimization
    % Inputs:
    % x - Input signal
    % t - Time vector
    % N - Number of samples
    % T - Signal period
    % K - Number of harmonics to consider
    % Outputs:
    % reconstructed_signal - Reconstructed signal
    % tk_best, rk_best, ck_best - Optimized delay, pulse width, and amplitude parameters

    % Fourier coefficient parameters
    Kmax = 4 * K + 2; % Maximum number of Fourier components
    m = -Kmax:Kmax; % Frequency indices

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
    y = real(y) / N; % Normalize reconstructed signal

    % TEM parameters
    b = 1.75; % TEM bias
    d = 0.08; % TEM threshold
    kappa = 0.5; % TEM scaling factor

    % Perform TEM sampling
    [tnIdx, ~] = iafTEM(y, T / N, b, d, kappa); % Obtain firing times
    tn = t(tnIdx); % Convert indices to time values
    yDel = -b * diff(tn) + kappa * d; % Differential delays

    % Fourier sample recovery
    w0 = 2 * pi / T; % Fundamental frequency
    F = exp(1j * w0 * tn(2:end)' * (-Kmax:Kmax)) - exp(1j * w0 * tn(1:end-1)' * (-Kmax:Kmax)); % Fourier basis
    F(:, Kmax+1) = tn(2:end) - tn(1:end-1); % Add temporal differences
    s = T ./ (1j * 2 * pi * (-Kmax:Kmax)); % Scale factor for Fourier coefficients
    s(Kmax+1) = 1;
    S = diag(s); % Diagonal scaling matrix

    % Solve for Fourier coefficients
    ytnHat = pinv(F * S) * yDel'; % Solve system of equations for Fourier coefficients
    ytnHat = ytnHat' * N; % Normalize coefficients

    % Spectrum processing
    spectrum = conj(ytnHat(Kmax+1:end)); % Extract positive frequency spectrum
    spectrum = cadzow(spectrum, Kmax-2, inf)'; % Apply Cadzow denoising

    % Parameter estimation
    [tk, rk, ck] = estimate_parameters(spectrum, T, N, K);

    % Optimize using pointwise error minimization
    params_init = [ck, tk, rk];
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
        'Display', 'off', 'MaxIterations', 1000, 'OptimalityTolerance', 1e-6);
    objective = @(params) pointwise_error(params, t, x, K);
    best_params = fminunc(objective, params_init, options);

    % Extract optimized parameters
    ck_best = best_params(1:K);
    tk_best = best_params(K+1:2*K);
    rk_best = best_params(2*K+1:end);

    % Reconstruct the signal with optimized parameters
    reconstructed_signal = zeros(size(t));
    for k = 1:K
        reconstructed_signal = reconstructed_signal + time_eval(ck_best, rk_best, tk_best, t, k);
    end
end

function [tk, rk, ck] = estimate_parameters(spectrum, T, N, K)
    % Function to estimate parameters tk, rk, and ck from spectrum
    l = round(length(spectrum) / 2) * 2; % Adjust length
    tr = flip(spectrum(1:(length(spectrum)) / 2 + 2)); % First half of spectrum (flipped)
    tc = spectrum((length(spectrum)) / 2 + 2:end); % Second half of spectrum
    tt = toeplitz(tc, tr); % Construct Toeplitz matrix

    % Perform SVD
    [~, ~, V] = svd(tt);
    V = conj(V(:, 1:K))'; % Take first K singular vectors
    V(1,:) = -V(1,:); % Adjust sign convention
    V = V'; % Transpose for computation
    v1 = V(1:end-1,:); % Submatrix 1
    v2 = V(2:end,:); % Submatrix 2

    % Eigenvalue decomposition
    [~, w] = eig(pinv(v2) * v1);
    w = conj(w); % Conjugate eigenvalues
    uk = diag(w)'; % Eigenvalues (roots of annihilating filter)

    % Compute delays and pulse widths
    tk = T * atan2(imag(uk), real(uk)) / (2 * pi); % Delays
    rk = -T * log(abs(uk)) / (2 * pi); % Pulse widths
    rk(rk <= 0) = T / N; % Assign minimum pulse width

    % Estimate amplitudes
    ck = 1 / T * pinv(vander2(uk, length(spectrum)))' * spectrum';
    tk = mod(tk, T); % Wrap delays to [0, T]
    rk = sort(rk, 'descend'); % Sort pulse widths
    ck = abs(ck') / N; % Normalize amplitudes
end
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
    
    % Compute pointwise error as sum of absolute differences
    error = mean(abs(x - signal2));
end