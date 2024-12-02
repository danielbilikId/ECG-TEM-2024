function reconstructed_signal = reconstructECG(tnIdx, y_out, b, k, d, dt)
    % Reconstruct ECG signal from iafTEM outputs
    %
    % INPUT:  Time stamp indices, tnIdx
    %         Integrator output, y_out
    %         Bias, b
    %         Integrator constant, k
    %         Threshold, d
    %         Time resolution, dt
    %
    % OUTPUT: Reconstructed ECG signal
    
        % Compute the partial summations for reconstruction
        yn = -b * diff([0 tnIdx]) + k * d;
    
        % Reconstruct the Fourier coefficients
        % Assuming Fourier series computation for VPW-FRI pulses
        M = length(yn); % Number of Fourier coefficients
        f_series = zeros(1, M); % Fourier series coefficients
    
        for m = 1:M
            f_series(m) = sum(yn .* exp(-1j * 2 * pi * m * (tnIdx / dt)));
        end
    
        % Reconstruct signal using Fourier coefficients
        t = linspace(0, length(y_out) * dt, length(y_out));
        reconstructed_signal = real(ifft(f_series, length(t)));
    end
    