%% load data
clear; clc;
load_directory

n_trials = 50; % Number of trials
alpha = 0.3; % percentage of outliers
m = 30; 
show_output = 0;
max_iter = 100;

p = 500;  % number of points
d = 2;    % dimension of the points
r = d + 2; 

rmse_all = zeros(n_trials, 1); 

for i = 1:n_trials
    % [D_obs, X_true, P_true, D_true] = generate_data_RDMS(alpha, p, d); % Generate data
    [F_obs, E_true, F_true, P_true] = generate_data_SREDG(alpha, m, p, d);

    data = struct( ...
        'F_obs', F_obs, ... 
        'E_true', E_true, ...
        'P_true', P_true, ...
        'F_true', F_true ...
    );


    params.show_output = show_output; 
    params.max_iter = max_iter; 
    params.d = d; % dimension of the points
    params.alpha = alpha;
    params.known_row_id = 1;


    [~, P_estimated, ~ ] = SREDGAAP(data, params);

    %% Compute RMSE
    [rmse, ~, ~] = Compute_RMSE(P_true',P_estimated);
    rmse_all(i) = rmse;

    fprintf('Trial %d/%d: RMSE = %.4g\n', i, n_trials, rmse);



    fprintf('p = %d,  alpha = %f, RMSE = %.4g\n', p, alpha, rmse);
    fprintf('================================\n');

end

% Compute and display the mean RMSE
mean_rmse = mean(rmse_all);
recovered = sum(rmse_all < 0.1); % count the number of trials with RMSE < 0.1
fprintf('================================\n');
fprintf('p = %d, alpha = %.2f, trial = %d, mean RMSE = %.4g, recovered = %d\n', ...
        p, alpha, n_trials, mean_rmse, recovered);
