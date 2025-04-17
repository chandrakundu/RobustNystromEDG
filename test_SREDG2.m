
% This script is for sensor localization synthetic data. 

%% load data
clear; clc;
addpath LIB\
addpath rpca\

p = 500;  % number of points
d = 2;    % dimension of the points
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));

alpha = 0.3; % percentage of outliers
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


% nystrom on D 
D_estimated = [E F_corrupted; F_corrupted' G];
X_estimated_dist = dist2gram(D_estimated); % corrupted Gram matrix
P_estimated_dist = gram_to_points(X_estimated_dist, d); % estimated points from corrupted Gram matrix
[rmse_dist, ~, ~] = Compute_RMSE(P', P_estimated_dist); % RMSE of the estimated points
fprintf('RMSE (Nystrom on D): %f\n', rmse_dist);


% nystrom on X
[A, B] = compute_AB(E, F_corrupted); % compute blocks A and B of the Gram matrix
C = B'*pinv(A, 0.01)*B; % compute the Gram matrix
X_estimated_gram = [A B; B' C]; % estimated Gram matrix
P_estimated_gram = gram_to_points(X_estimated_gram, d); % estimated points from corrupted Gram matrix
[rmse_gram, ~, ~] = Compute_RMSE(P', P_estimated_gram); % RMSE of the estimated points
fprintf('RMSE (Nystrom on X): %f\n', rmse_gram);

%% ROBUST PCA Params
para.mu        = 1.1*get_mu_kappa(F,r);  
para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
para.trimming  = false;
para.tol       = 1e-14;
para.gamma     = 0.9;
para.max_iter  = 500;
para.show_output = 0;


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


function [A, B] = compute_AB(E, F)
    % COMPUTE_AB Computes blocks A and B of the Gram matrix from E and F blocks of distance matrices

    % Dimensions of E and F
    m = size(E, 1);
    n = size(F, 2);

    % Vector of ones
    ones_m = ones(m, 1);
    ones_n = ones(n, 1);

    ones_mm = (1/m) * (ones_m * ones_m');
    ones_mn = (1/m) * (ones_m * ones_n');

    A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
    B = -0.5 * (F - ones_mm * F - E * ones_mn + m*mean(E(:)) * ones_mn);
end

function X = dist2gram(D)
    p = size(D, 1);
    s = zeros(p,1);
    s(1:end) = 1/p;
    J = eye(p) - (ones(p,1)*s');
    X = -0.5*J*D*J';
end

function X_new = fix_gram_matrix(X, tolerance)
    % FIX_GRAM_MATRIX Fixes the negative eigenvalues of the Gram matrix and ensures symmetry
    %
    % Input:
    % X: (m+n) x (m+n) matrix, the Gram matrix
    % tolerance: the tolerance for fixing the negative eigenvalues
    
    if nargin < 2
        tolerance = 0; 
    end
    
    % Eigen decomposition
    [V, D] = eig(X);
    
    % set those smaller than the tolerance to zero
    D = diag(D); 
    D(D < tolerance) = 0; 
    D = diag(D); 
    
    % Reconstruct
    X_new = V * D * V';

    % Ensure symmetry and real values
    X_new = (X_new + X_new') / 2; 
    X_new = real(X_new);
end

function P = gram_to_points(X, d)
    [V, Lam] = eigs(X, d, 'lm');
    P = V * sqrt(Lam);
end
