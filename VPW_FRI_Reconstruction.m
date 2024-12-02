function [t_est, r_est, c_est] = VPW_FRI_Reconstruction(samples, N, T, K)
    % VPW-FRI Reconstruction Algorithm
    % Inputs:
    %   samples - sampled data points
    %   N       - number of samples
    %   T       - sampling period
    %   K       - number of pulses
    % Outputs:
    %   t_est - estimated pulse locations
    %   r_est - estimated pulse widths
    %   c_est - estimated pulse amplitudes

    % Compute the Fourier coefficients from the samples
    G = fft(samples) / N;
    G = G(1:(4*K+1)); % Take 2K+1 coefficients for annihilation
    G = G(:); % Ensure G is a column vector

    % Construct the Toeplitz matrix for the annihilating filter
    S = toeplitz(G(K+1:end), G(K+1:-1:1));
    
    % Solve for annihilating filter coefficients
    [~, ~, V] = svd(S);
    h = V(:, end); % Null space of S gives annihilating filter

    % Compute the roots of the annihilating filter
    roots_h = roots(h);
    roots_h = roots_h(abs(roots_h) < 1); % Filter unstable roots


    % Extract pulse locations (t_est) and widths (r_est)
    t_est = -T * angle(roots_h) / (2 * pi);
    r_est = -T * log(abs(roots_h)) / (2 * pi);

    % Ensure roots align with available Fourier coefficients
    if length(roots_h) ~= K
        error('Mismatch between number of roots and K. Check input signal and parameters.');
    end

    % Construct Vandermonde matrix and solve for amplitudes
    V_matrix = zeros(length(G), K);
    for m = 1:length(G)
        V_matrix(m, :) = roots_h(:).'.^(m-1);
    end

    % Solve for amplitudes
    c_est = V_matrix \ G; % Least squares solution

    % Ensure locations are within one period
    t_est = mod(t_est, T); 
end
