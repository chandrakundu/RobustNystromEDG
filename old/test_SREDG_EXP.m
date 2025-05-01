%% load data
clear; clc;
load_directory


n_trials = 1; % Number of trials
alpha = 0.1; % percentage of outliers
m = 40;   % number of anchors, here minimum m = round(4*(d+2)*log(p));
show_output = 0;

p = 500;  % number of points
d = 3;    % dimension of the points
r = d + 2; 
n = p - m; 


rmse_all = zeros(n_trials, 1); 

for i = 1:n_trials
    [E_true, F_corrupted, P_true, ~, F_true, ~] = generate_data(alpha, m, p, d);

    data = struct( ...
        'E_true', E_true, ...
        'F_corrupted', F_corrupted, ...
        'P_true', P_true, ...
        'F_true', F_true ...
    );


    params.show_output = show_output; 
    params.d = d; % dimension of the points


    [~, P_estimated, ~ ] = SREDG_EXP(data, params);

    %% Compute RMSE
    [rmse, ~, ~] = Compute_RMSE(P_true',P_estimated);
    rmse_all(i) = rmse;

    fprintf('Trial %d/%d: RMSE = %.4g\n', i, n_trials, rmse);



    fprintf('p = %d, m = %d, alpha = %f, RMSE = %.4g\n', p, m, alpha, rmse);
    fprintf('================================\n');

end

% Compute and display the mean RMSE
mean_rmse = mean(rmse_all);
recovered = sum(rmse_all < 0.1); % count the number of trials with RMSE < 0.1
fprintf('================================\n');
fprintf('p = %d, m = %d, alpha = %.2f, trial = %d, mean RMSE = %.4g, recovered = %d\n', ...
        p, m, alpha, n_trials, mean_rmse, recovered);

%% Visualization
% plot_points(P',P_estimated,m)
