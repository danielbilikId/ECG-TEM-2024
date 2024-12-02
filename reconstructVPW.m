function reconstructed_signal = reconstructVPW(Tk, rk, ck, dk, T, N, dt)
% Reconstruct VPW-FRI signal from parameters
%
% INPUT:
% Tk    - Temporal delays
% rk    - Pulse widths
% ck    - Symmetric amplitudes
% dk    - Asymmetric amplitudes
% T     - Signal period
% N     - Length of the output signal
% dt    - Time resolution
%
% OUTPUT:
% reconstructed_signal - Reconstructed VPW signal

    t = (0:N-1) * dt;
    reconstructed_signal = zeros(1, N);

    for k = 1:length(Tk)
        pulse = (ck(k) ./ (pi * (rk(k)^2 + (t - Tk(k)).^2))) + ...
                (dk(k) .* (t - Tk(k)) ./ (pi * (rk(k)^2 + (t - Tk(k)).^2)));
        reconstructed_signal = reconstructed_signal + pulse;
    end
end
