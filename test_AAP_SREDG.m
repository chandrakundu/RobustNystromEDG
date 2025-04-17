
% This script is for sensor localization synthetic data. 

%% load data
clear; clc;
addpath LIB\
addpath rpca\

p = 500;  % number of points
d = 3;    % dimension of the points
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));

alpha = 0.2; % percentage of outliers
n = p - m; % number of sensors

% s = rng('shuffle');
% sd = 183252468;alpha=0.3;m=20;p=500;d=3; % very  bad problem seed
% sd = 'shuffle';
% s = rng(sd);
P1 = -100+200.*rand(d,m); 
P2 = -100+200.*rand(d,n); 

P = [P1 P2];  
P = P - mean(P,2); % center the points


% Gram matrix
X = P'*P; 

% Squared distance matrix
dist = squareform(pdist(P'));
D = dist.*dist;
r = d + 2; % rank of the distance matrix

% Blocks of D
E = D(1:m,1:m);
F = D(1:m,m+1:end);
G = D(m+1:end,m+1:end);
F_corrupted = get_sparse_noise(F, alpha); % added sparse noise


%% Apply AAP_SREDG
known_row_id = m;
known_row_F = F(known_row_id,:);  
tol = 1e-14;
zeta0 = 1 * max(F(:));
gamma = 0.9;
max_iter = 50;
accelerated = true; 
show_output = 2;
[X_estimated, P_estimated, ~] = SREDG_AAP_EXP(E, F_corrupted, F, X, P, d, known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol, show_output);
% [X_estimated, P_estimated, ~] = AAP_SREDGRPCA(E, F_corrupted, X, P, d, known_row_F, known_row_id, zeta0, gamma, max_iter, tol, show_output);


%% Compute RMSE
[rmse, ~, ~] = Compute_RMSE(P',P_estimated);
fprintf('p = %d, m = %d, alpha = %f, RMSE = %.4g\n', p, m, alpha, rmse);
fprintf('================================\n');



%% Visualization
% plot_points(P',P_estimated,m)



%% helper functions

function F_corrupted = get_sparse_noise(F, alpha)
    m = size(F, 1);
    n = size(F, 2);
    S_supp_idx = randsample(m*n, round(alpha*m*n), false);
    S_range = 1*mean(mean(abs(F)));
    S_temp = 2*S_range*rand(m,n)-S_range; 
    S_true = zeros(m, n);
    S_true(S_supp_idx) = S_temp(S_supp_idx);  
    F_corrupted = F + S_true;
end
