% test_geometric_cleanup.m
% Test GeometricConsistencyCleanupCosine on synthetic data
clc; clear; close all;
load_directory;
rng(42); % For reproducibility

% Parameters
d = 3;      % dimension
m = 10;     % number of anchors
n = 20;     % number of targets
alpha = 0.1; % fraction of corrupted entries in F

% Generate random points
P1 = randn(d, m);
P2 = randn(d, n);
P = [P1, P2];

% Compute squared distance matrix D
D = pdist2(P', P').^2;

% Partition D
E = D(1:m, 1:m);
F = D(1:m, m+1:end);
G = D(m+1:end, m+1:end);

% Add sparse noise to F to get F_hat
F_hat = F;
num_corrupt = round(alpha * numel(F));
idx = randperm(numel(F), num_corrupt);
noise = 10 * std(F(:)) * (2*rand(size(idx))-1); % large noise
F_hat(idx) = F_hat(idx) + noise;

% Clean F_hat
F_clean = GeometricConsistencyCleanup(E, F_hat);

% Compute RMSEs
rmse_noisy = sqrt(mean((F_hat(:) - F(:)).^2));
rmse_clean = sqrt(mean((F_clean(:) - F(:)).^2));

fprintf('RMSE (noisy F_hat vs F):  %.4f\n', rmse_noisy);
fprintf('RMSE (cleaned F_clean vs F): %.4f\n', rmse_clean);

if rmse_clean < rmse_noisy
    disp('GeometricConsistencyCleanup improved the result.');
else
    disp('No improvement from GeometricConsistencyCleanup.');
end
