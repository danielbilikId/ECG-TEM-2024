% Load ECG signal
signal = load("C:\Users\danie\workspace\ecg\GDN0001\GDN0001_2_Valsalva.mat"); 
x = signal.tfm_ecg2(1:1600); % Extract signal as column vector
x = x / max(x); % Normalize the signal to [0, 1]

% Generate time vector
N = length(x); % Number of samples
T = 1; % Signal period
dt = 1 / N; % Time step
t = linspace(0, T, N); % Time vector
t_eval = linspace(0, T, N); % Evaluation time vector

% Parameters (replace with your extracted FRI parameters)
% Example values: replace these with your actual FRI parameters
tk = [0.8, 0.6, 0.4, 0.1, 0.95];
ck = [1, -0.3, 0.8, -0.5, 0.5];
rk = [0.1, 0.08, 0.05, 0.06, 0.09];
K = length(tk);

% Generate all permutations of tk, ck, rk
perm_idx = perms(1:K); % Permutations of indices
num_perms = size(perm_idx, 1);

% Initialize variables to store the best result
best_signal = [];
best_mse = inf;
best_perm = [];

% Loop through all permutations
for i = 1:num_perms
    % Get the current permutation of indices
    idx = perm_idx(i, :);
    
    % Reorder FRI parameters
    tk_perm = tk(idx);
    ck_perm = ck(idx);
    rk_perm = rk(idx);
    
    % Reconstruct signal with current permutation
    signal2 = zeros(size(t_eval));
    for k = 1:K
        signal2 = signal2 + time_eval(ck_perm, rk_perm, tk_perm, t_eval, k); % VPW evaluation
    end
    
    % Fix flipped polarity (if needed)
    for k = 1:K
        % Check correlation with original signal
        if corr(signal2, x') < 0
            ck_perm(k) = -ck_perm(k); % Flip the polarity of the amplitude
        end
        signal2 = signal2 + time_eval(ck_perm, rk_perm, tk_perm, t_eval, k); % VPW evaluation
    end
    % Normalize reconstructed signal
    signal2 = signal2 / max(signal2);
    
    % Compute MSE with the original signal
    mse = mean((x - signal2).^2);
    
    % Update best signal if current MSE is better
    if mse < best_mse
        best_mse = mse;
        best_signal = signal2;
        best_perm = idx;
    end
end


% Reorder FRI parameters based on time indices
[tk_sorted, idx] = sort(tk); % Sort time indices
ck_sorted = ck(idx); % Reorder amplitudes
rk_sorted = rk(idx); % Reorder widths

% Fix flipped polarity (if needed)
for k = 1:K
    % Evaluate the VPW pulse at each k
    vpw_signal = time_eval(ck_sorted, rk_sorted, tk_sorted, t_eval, k);
    % Check correlation with original signal
    if corr(vpw_signal, x') < 0
        ck_sorted(k) = -ck_sorted(k); % Flip the polarity of the amplitude
    end
    signal2 = signal2 + time_eval(ck_sorted, rk_sorted, tk_sorted, t_eval, k); % VPW evaluation
end

% ================================ Section 6: Visualization =======================================
% Plot original and best reconstructed signals
figure;
plot(t, x, 'k', 'LineWidth', 1.5); hold on;
plot(t, best_signal, '--r', 'LineWidth', 1.5);
legend('Original Signal', 'Best Reconstructed Signal');
xlabel('Time (s)');
ylabel('Amplitude');
title('Original vs. Best Reconstructed ECG Signal');
grid on;

% Display the best permutation and MSE
disp('Best permutation of indices:');
disp(best_perm);
disp(['Best MSE: ', num2str(best_mse)]);

