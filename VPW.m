% Step 1: Generate Synthetic Pulse Train
fs = 1000; % Sampling frequency
t = 0:1/fs:1; % Time vector
num_pulses = 5;
amplitudes = rand(1, num_pulses); % Random amplitudes
widths = 0.02 + 0.03 * rand(1, num_pulses); % Random widths
locations = sort(rand(1, num_pulses) * max(t)); % Random sorted locations

% Generate pulse train
x = zeros(size(t));
for k = 1:num_pulses
    pulse = amplitudes(k) * exp(-((t - locations(k)) / widths(k)).^2); % Gaussian pulse
    x = x + pulse;
end

% Add noise
noise_std = 0.1;
y = x + noise_std * randn(size(t));

% Step 2: Sampling
sample_rate = 50; % Downsample rate
t_sampled = t(1:fs/sample_rate:end);
y_sampled = y(1:fs/sample_rate:end);

% Step 3: Define Pulse Model
pulse_model = @(t, a, loc, w) a * exp(-((t - loc) / w).^2);

% Step 4: Optimization Problem (Least Squares Fit)
params = []; % Initialize pulse parameters
for k = 1:num_pulses
    % Initial guesses
    a0 = max(y_sampled);
    loc0 = locations(k);
    w0 = widths(k);
    
    % Objective function for optimization
    obj_fun = @(p) norm(y_sampled - pulse_model(t_sampled, p(1), p(2), p(3)));
    % Optimize using fminsearch
    params_k = fminsearch(obj_fun, [a0, loc0, w0]);
    params = [params; params_k];
end

% Step 5: Reconstruct Signal
x_reconstructed = zeros(size(t));
for k = 1:size(params, 1)
    x_reconstructed = x_reconstructed + pulse_model(t, params(k, 1), params(k, 2), params(k, 3));
end

% Plot Results
figure;
subplot(3, 1, 1); plot(t, x, 'LineWidth', 1.5); title('Original Signal');
subplot(3, 1, 2); plot(t, y, 'LineWidth', 1.5); title('Noisy Signal');
subplot(3, 1, 3); plot(t, x_reconstructed, 'LineWidth', 1.5); title('Reconstructed Signal');
