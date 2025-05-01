%% load data
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
[E, F_corrupted, P_true, ~, F, G_true] = generate_data(alpha, m, p, d);

A = compute_A(E);
B = compute_B(E, F);

k = 5;
known_row_F = F(k, :); % known row of F
% F2 = compute_F(B, E, known_row_F, k);
F2 = computeF2(B, E);


err_F = norm(F2 - F)/norm(F);
fprintf('err_F = %.4g\n', err_F);
