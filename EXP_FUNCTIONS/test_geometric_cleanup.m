%% Test script to compare geometric consistency cleanup methods
% This script compares two methods for cleaning noisy anchor-target 
% squared distance matrices: 
% 1. GeometricConsistencyCleanupCosine (law-of-cosines based)
% 2. GeometricConsistencyCleanupShortestPath (shortest-path based)

%% Parameters
clear; clc; close all;
% load_directory;
rng(42); % For reproducibility

% Data generation parameters
d = 3;      % dimension of points
m = 20;     % number of anchor points
n = 500;     % number of target points
T = m + n;  % total number of points

% Noise parameters
alpha = 0.1;  % fraction of corrupted entries in F
noise_scale = 5;  % scale of noise relative to standard deviation

%% Generate synthetic data
% Random points in R^d
P1 = randn(d, m);  % anchor points
P2 = randn(d, n);  % target points
P = [P1, P2];      % all points

% Gram matrix X = P'*P
A = P1' * P1;
B = P1' * P2;
C = P2' * P2;
X = [A, B; B', C];

% Squared distance matrix D
D = zeros(T, T);
for i = 1:T
    for j = 1:T
        D(i,j) = norm(P(:,i) - P(:,j))^2;
    end
end

% Partition D into E, F, G
E = D(1:m, 1:m);          % anchor-anchor
F = D(1:m, m+1:end);      % anchor-target
G = D(m+1:end, m+1:end);  % target-target

%% Add sparse noise to F
F_hat = F;
num_corrupt = round(alpha * m * n);
[idx_i, idx_j] = ind2sub([m, n], randperm(m*n, num_corrupt));

% Generate large noise (both positive and negative)
std_F = std(F(:));
noise = noise_scale * std_F * (2*rand(size(idx_i))-1);

% Apply noise
for k = 1:num_corrupt
    F_hat(idx_i(k), idx_j(k)) = F_hat(idx_i(k), idx_j(k)) + noise(k);
end

%% Clean F_hat using both methods
tic;
F_clean_cosine = GeometricConsistencyCleanupCosine(E, F_hat);
time_cosine = toc;

tic;
F_clean_shortest = GeometricConsistencyCleanupShortestPath(E, F_hat);
time_shortest = toc;

%% Evaluate results
% Compute RMSEs
rmse_noisy = sqrt(mean((F_hat(:) - F(:)).^2));
rmse_cosine = sqrt(mean((F_clean_cosine(:) - F(:)).^2));
rmse_shortest = sqrt(mean((F_clean_shortest(:) - F(:)).^2));

% Compute relative errors
rel_err_noisy = norm(F_hat - F, 'fro') / norm(F, 'fro');
rel_err_cosine = norm(F_clean_cosine - F, 'fro') / norm(F, 'fro');
rel_err_shortest = norm(F_clean_shortest - F, 'fro') / norm(F, 'fro');

% Compute percentage improvement
improve_cosine = (rmse_noisy - rmse_cosine) / rmse_noisy * 100;
improve_shortest = (rmse_noisy - rmse_shortest) / rmse_noisy * 100;

%% Display results
fprintf('=== Test Results ===\n');
fprintf('Data Parameters: d=%d, m=%d, n=%d, alpha=%.2f\n\n', d, m, n, alpha);

fprintf('RMSE:\n');
fprintf('  Noisy F_hat:        %.6f\n', rmse_noisy);
fprintf('  Cosine cleanup:     %.6f (%.2f%% improvement)\n', rmse_cosine, improve_cosine);
fprintf('  Shortest-path:      %.6f (%.2f%% improvement)\n', rmse_shortest, improve_shortest);

fprintf('\nRelative Error:\n');
fprintf('  Noisy F_hat:        %.6f\n', rel_err_noisy);
fprintf('  Cosine cleanup:     %.6f\n', rel_err_cosine);
fprintf('  Shortest-path:      %.6f\n', rel_err_shortest);

fprintf('\nComputation Time:\n');
fprintf('  Cosine cleanup:     %.4f seconds\n', time_cosine);
fprintf('  Shortest-path:      %.4f seconds\n', time_shortest);

%% Visualization
% Number of corrupted vs fixed entries
fixed_cosine = 0;
fixed_shortest = 0;

for k = 1:num_corrupt
    i = idx_i(k);
    j = idx_j(k);
    
    % Check if methods improved the entry (closer to true value)
    if abs(F_clean_cosine(i,j) - F(i,j)) < abs(F_hat(i,j) - F(i,j))
        fixed_cosine = fixed_cosine + 1;
    end
    
    if abs(F_clean_shortest(i,j) - F(i,j)) < abs(F_hat(i,j) - F(i,j))
        fixed_shortest = fixed_shortest + 1;
    end
end

fprintf('\nCorrupted entries fixed:\n');
fprintf('  Cosine cleanup:     %d/%d (%.2f%%)\n', fixed_cosine, num_corrupt, 100*fixed_cosine/num_corrupt);
fprintf('  Shortest-path:      %d/%d (%.2f%%)\n', fixed_shortest, num_corrupt, 100*fixed_shortest/num_corrupt);

% Plot error comparison for corrupted entries
figure;
subplot(1,2,1);
scatter(F(idx_i, idx_j), F_hat(idx_i, idx_j), 'b.');
hold on;
plot([min(F(:)), max(F(:))], [min(F(:)), max(F(:))], 'r--');
xlabel('True F');
ylabel('Corrupted F\_hat');
title('Corrupted Values');
axis square;

subplot(1,2,2);
scatter(F(idx_i, idx_j), F_clean_cosine(idx_i, idx_j), 'b.', 'DisplayName', 'Cosine');
hold on;
scatter(F(idx_i, idx_j), F_clean_shortest(idx_i, idx_j), 'r.', 'DisplayName', 'Shortest Path');
plot([min(F(:)), max(F(:))], [min(F(:)), max(F(:))], 'k--');
xlabel('True F');
ylabel('Cleaned F');
title('Cleaned Values');
legend('Location', 'best');
axis square;

% Plot heat maps of errors
figure;
subplot(1,3,1);
imagesc(abs(F_hat - F));
colorbar;
title('Error in F\_hat');
axis square;

subplot(1,3,2);
imagesc(abs(F_clean_cosine - F));
colorbar;
title('Error after Cosine Cleanup');
axis square;

subplot(1,3,3);
imagesc(abs(F_clean_shortest - F));
colorbar;
title('Error after Shortest Path Cleanup');
axis square;
