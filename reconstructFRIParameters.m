function [Tk, rk, ck, dk] = reconstructFRIParameters(tn, b, k, d, T, K)
% Reconstruct FRI parameters from IF-TEM spike times
%
% INPUT:
% tn    - Spike times
% b     - Bias
% k     - Integrator constant
% d     - Threshold
% T     - Signal period
% K     - Number of VPW-FRI pulses
%
% OUTPUT:
% Tk    - Temporal delays
% rk    - Pulse widths
% ck    - Symmetric amplitudes
% dk    - Asymmetric amplitudes

    % Step 1: Compute yn
    yn = -b * diff(tn) + k * d; % Compute yn based on consecutive tn differences

    % Step 2: Create Vandermonde matrix B
    omega0 = 2 * pi / T;
    M = 6 * K; % Number of Fourier coefficients
    N = length(yn); % Correct number of rows matches yn length
    B = zeros(N, 2*M+1);
    for i = 1:N
        for m = -M:M
            B(i, m+M+1) = exp(1j * m * omega0 * tn(i));
        end
    end

    % Step 3: Solve for Fourier coefficients z_hat
    z_hat = pinv(B) * yn.'; % Correct dimensions for multiplication

    % Step 4: Remove the zero-frequency component (m = 0)
    z_hat(M+1) = 0; % Set the 0th frequency component to zero

    % Step 5: Extract FSCs for positive indices
    x_hat = z_hat(M+2:end); % Positive frequency components only

    % Step 6: Construct the annihilating filter
    A = zeros(K+1, 1);
    for i = 1:K
        A(i) = x_hat(i) / i; % Example computation for filter coefficients
    end

   % Find roots and filter only stable ones (inside unit circle)
    uk = roots(A);
    uk = uk(abs(uk) < 1); % Retain roots inside the unit circle


    % Step 8: Extract FRI parameters from roots
    Tk = -T * angle(uk) / (2 * pi); % Temporal delays
    rk = T * log(abs(uk)) / (2 * pi); % Pulse widths

    % Step 9: Compute amplitudes
    ck = real(x_hat(1:K)); % Symmetric amplitudes
    dk = imag(x_hat(1:K)); % Asymmetric amplitudes
end
