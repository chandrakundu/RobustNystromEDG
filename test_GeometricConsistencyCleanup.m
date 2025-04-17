%% load data
clear; clc;
load_directory


n_trials = 50; % Number of trials
alpha = 0.3; % percentage of outliers
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));

p = 500;  % number of points
d = 3;    % dimension of the points
r = d + 2; 
n = p - m; 


rmse_all = zeros(n_trials, 1); 
rmse_all_corrupted = zeros(n_trials, 1);

for i = 1:n_trials
    [E_true, F_corrupted, P_true, ~, F_true, ~] = generate_data(alpha, m, p, d);

    data = struct( ...
        'E_true', E_true, ...
        'F_corrupted', F_corrupted, ...
        'P_true', P_true, ...
        'F_true', F_true ...
    );

    F_clean = GeometricConsistencyCleanup(E_true, F_corrupted);

    err_F_corrupted = norm(F_corrupted - F_true, 'fro') / norm(F_true, 'fro');
    err_F_clean = norm(F_clean - F_true, 'fro') / norm(F_true, 'fro');

    rmse_all(i) = err_F_clean;
    rmse_all_corrupted(i) = err_F_corrupted;

    fprintf('Trial %d/%d: F_corrupted RMSE = %.4g, F_clean RMSE = %.4g\n', i, n_trials, err_F_corrupted, err_F_clean);


    params.show_output = 2; 
    params.d = d; % dimension of the points
    fprintf('================================\n');

end

% Compute and display the mean RMSE
mean_rmse = mean(rmse_all);
corrupted_mean_rmse = mean(rmse_all_corrupted);
fprintf('================================\n');
fprintf('p = %d, m = %d, alpha = %.2f, trial = %d, Corrupted mean RMSE = %.4g,  Clean mean RMSE = %.4g\n', ...
        p, m, alpha, n_trials,corrupted_mean_rmse, mean_rmse);

%% Visualization
% plot_points(P',P_estimated,m)
