clear; clc;
load_directory


n_trials = 1; % Number of trials
alpha = 0.1; % percentage of outliers
m = 20;   % number of anchors, here minimum m = round(4*(d+2)*log(p));
show_output = 0;

p = 500;  % number of points
d = 3;    % dimension of the points
r = d + 2; 
n = p - m; 


[E_true, F_corrupted, P_true, ~, F_true, ~] = generate_data(alpha, m, p, d);

data = struct( ...
    'E_true', E_true, ...
    'F_corrupted', F_corrupted, ...
    'P_true', P_true, ...
    'F_true', F_true ...
);

maxF = max(F_true(:));
kappa = cond(F_true);
mu12 = get_mu_kappa(F_true,r);
mu = sqrt(mu12(1)*mu12(2));

gamma = 0.9;

fprintf('maxF = %.4g, kappa = %.4g, mu = %.4g and %.4g\n', maxF, kappa, mu12(1), mu12(2));


alpha_theoretical = gamma / (1624 * mu * r * kappa^2) 
