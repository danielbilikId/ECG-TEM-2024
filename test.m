% Load and normalize the signal
signal = load("C:\Users\danie\workspace\ecg\GDN0001\GDN0001_2_Valsalva.mat"); 
t_eval = linspace(0, 1, 1600); % Time vector
x = signal.tfm_ecg2(1:1600); % Original signal
x = x / max(x); % Normalize signal

% Initial FRI parameters (example values, replace with actual parameters)
ck = [1, -0.5, 0.8, -0.3, 0.5]; % Amplitudes
tk = [0.1, 0.4, 0.6, 0.8, 0.95]; % Time indices
rk = [0.05, 0.08, 0.1, 0.06, 0.09]; % Widths

% Ensure the signal is normalized
x = x / max(x);

% Number of pulses
K = length(ck);

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
signal2 = signal2; % Normalize reconstructed signal

% Plot original and reconstructed signals
figure;
plot(t_eval, x, 'k', 'LineWidth', 1.5); hold on;
plot(t_eval, signal2, '--r', 'LineWidth', 1.5);
legend('Original Signal', 'Best Reconstructed Signal');
xlabel('Time (s)');
ylabel('Amplitude');
title('Original vs. Best Reconstructed ECG Signal');
grid on;

% Display optimized parameters
disp('Optimized Parameters:');
disp('ck (amplitudes):');
disp(ck_best);
disp('tk (time indices):');
disp(tk_best);
disp('rk (widths):');
disp(rk_best);

% ================================ Helper Functions =======================================

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

function vpw = time_eval(ck, rk, tk, t_eval, k)
    % Generate VPW pulse for kth pulse
    % Inputs:
    % ck, rk, tk - Pulse parameters
    % t_eval     - Evaluation time vector
    % k          - Current pulse index
    % Output:
    % vpw        - Evaluated VPW pulse
    vpw = ck(k) * exp(-((t_eval - tk(k)).^2) / (2 * rk(k)^2)); % Example Gaussian pulse
end
