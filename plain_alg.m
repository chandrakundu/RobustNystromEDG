%% load data
clear; clc;
addpath LIB\
addpath rpca\

p = 500;  % n.umber of points
d = 2;    % dimension of the points
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));
n = p - m; % number of sensors

alpha = 0.2; % percentage of outliers

[E, F_corrupted, P, Et, F, G] = generate_data(alpha, m, p, d, 'shuffle'); % generate data

% robust pca 
para.mu        = 1.1*get_mu_kappa(F,d);  
para.beta_init = d*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
para.beta      = d*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
para.trimming  = false;
para.tol       = 1e-14;
para.gamma     = 0.9;
para.max_iter  = 500;
para.show_output = 1;



% nystrom on D 
% D_estimated = [E F_corrupted; F_corrupted' G]; % corrupted distance matrix
G_estimated = F_corrupted'*pinv(E, 0.01)*F_corrupted; % compute the Gram matrix
D_estimated = [E F_corrupted; F_corrupted' G_estimated]; % corrupted distance matrix
X_estimated_dist = dist2gram(D_estimated, m); % corrupted Gram matrix
P_estimated_dist = gram_to_points(X_estimated_dist, d); % estimated points from corrupted Gram matrix
[rmse_dist, ~, ~] = Compute_RMSE(P', P_estimated_dist); % RMSE of the estimated points
fprintf('RMSE (Nystrom on D with corrupted F): %f\n', rmse_dist);

% nystrom on X
[A, B] = compute_AB(E, F_corrupted); % compute blocks A and B of the Gram matrix
C = B'*pinv(A, 0.01)*B; % compute the Gram matrix
X_estimated_gram = [A B; B' C]; % estimated Gram matrix
P_estimated_gram = gram_to_points(X_estimated_gram, d); % estimated points from corrupted Gram matrix
[rmse_gram, ~, ~] = Compute_RMSE(P', P_estimated_gram); % RMSE of the estimated points
fprintf('RMSE (Nystrom on X with corrupted F): %f\n', rmse_gram);

F_estimated = AccAltProj(F_corrupted, d+2, para); % robust PCA on F

G_estimated = F_estimated'*pinv(E, 0.01)*F_estimated; % compute the Gram matrix
D_estimated = [E F_estimated; F_estimated' G_estimated]; % corrupted distance matrix
X_estimated_dist = dist2gram(D_estimated, m); % corrupted Gram matrix
P_estimated_dist = gram_to_points(X_estimated_dist, d); % estimated points from corrupted Gram matrix
[rmse_dist, ~, ~] = Compute_RMSE(P', P_estimated_dist); % RMSE of the estimated points
fprintf('RMSE (Nystrom on D with RPCA): %f\n', rmse_dist);


% nystrom on X
[A, B] = compute_AB(E, F_estimated); % compute blocks A and B of the Gram matrix
C = B'*pinv(A, 0.01)*B; % compute the Gram matrix
X_estimated_gram = [A B; B' C]; % estimated Gram matrix
P_estimated_gram = gram_to_points(X_estimated_gram, d); % estimated points from corrupted Gram matrix
[rmse_gram, ~, ~] = Compute_RMSE(P', P_estimated_gram); % RMSE of the estimated points
fprintf('RMSE (Nystrom on X with RPCA): %f\n', rmse_gram);


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

function X = dist2gram(D, m)
    p = size(D, 1);
    s = zeros(p,1);
    s(1:m) = 1/m;
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


function [E_out, F_out, P_out, E_true, F_true, G_true] = generate_data(alpha, m, p, d, seed)
    % GENERATE_DATA Generates random sensor and anchor locations
    %
    % Inputs:
    % alpha: percentage of outliers
    % m: number of anchors
    % p: total number of points
    % d: dimension of the points
    % seed: random seed for reproducibility
    
    switch nargin
        case 0
            alpha = 0.1; % default value
            m = 30;      % default value
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 1
            m = 30;      % default value
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 2
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 3
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 4
            seed = 'shuffle'; % default value
    end

    rng(seed);  % Set the random seed for reproducibility
    
    % rng(1);
    P1 = -100 + 200 * rand(d, m); 
    P2 = -100 + 200 * rand(d, p - m); 
    
    P = [P1, P2];  
    P = P - mean(P, 2);  % Center the points
    
    % Compute squared distance matrix
    dist = squareform(pdist(P'));
    D = dist.^2;
    
    E = D(1:m, 1:m);
    F = D(1:m, m+1:end);
    if nargout > 2
        G = D(m+1:end, m+1:end); 
    end
     
    switch nargout
        case 1
            E_out = E;
        case 2
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
        case 3
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
        case 4
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
            E_true = E;
        case 5
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
            E_true = E;
            F_true = F;
        case 6
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
            E_true = E;
            F_true = F;
            G_true = G;
        otherwise
            error('Invalid number of output arguments. Expected 1 to 6 outputs.');
    end
end
