
% This script is for sensor localization synthetic data. 

%% load data
clear; clc;
addpath LIB\
addpath rpca\

p = 500;  % number of points
d = 2;    % dimension of the points
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));

alpha = 0.2; % percentage of outliers
n = p - m; % number of sensors

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


%% ROBUST PCA Params
para.mu        = 1.1*get_mu_kappa(F,r);  
para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
para.trimming  = false;
para.tol       = 1e-14;
para.gamma     = 0.9;
para.max_iter  = 500;
para.show_output = 2;
para.muB        = 1.1*get_mu_kappa(F,r);  

%% Apply SREDG
[X_estimated, P_estimated, ~ ] = SREDG(E,F_corrupted,r,@AccAltProj, para);


%% Compute RMSE
[rmse, ~, ~] = Compute_RMSE(P',P_estimated);
fprintf('p = %d, m = %d, alpha = %f, RMSE = %.4g\n', p, m, alpha, rmse);
fprintf('================================\n');



%% Visualization
plot_points(P',P_estimated,m)



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
