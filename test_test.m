%% load data
clear; clc;
load_directory


p = 500;  % number of points
d = 2;    % dimension of the points
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));
r = d + 2; % rank of the distance matrix

alpha = 0.2; % percentage of outliers
n = p - m; % number of sensors

[E, F_corrupted, P, E_true, F_true, G_true] = generate_data(alpha, m, p, d);


para = get_rpca_params(F_true, d+2);
para.show_output = 0; 


[X_estimated, P_estimated, ~ ] = SREDG(E,F_corrupted,r,@AccAltProj, para);

%% Compute RMSE
[rmse, ~, ~] = Compute_RMSE(P',P_estimated);
fprintf('p = %d, m = %d, alpha = %f, RMSE = %.4g\n', p, m, alpha, rmse);
fprintf('================================\n');



%% Visualization
plot_points(P',P_estimated,m)
