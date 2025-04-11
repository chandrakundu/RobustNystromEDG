[~, ~] = run_trials(50)

function [mean_rmse, rmse_all] = run_trials(nTrials)
% RUN_TRIALS runs sensor localization experiments for nTrials and 
% returns the mean RMSE over all trials.
%
% INPUT:
%   nTrials   - Number of trials to run.
%
% OUTPUT:
%   mean_rmse - Mean RMSE computed over all trials.
%   rmse_all  - A vector of RMSE values from each trial.
%
% Example:
%   [mean_rmse, rmse_all] = run_trials(10);
%
% Dependencies: Ensure that AAP_SREDG, Compute_RMSE, and LIB\ (with any
% necessary files) are on your MATLAB path.

    % Add required paths
    addpath('LIB\');

    % Set simulation parameters
    p = 500;       % total number of points
    d = 3;         % dimension of the points
    m = 30;        % number of anchors (minimum m = round(4*(d+2)*log(p)))
    alpha = 0.20;   % percentage of outliers
    n = p - m;     % number of sensors

    % Parameters for AAP_SREDG algorithm
    known_row_id = m;
    tol = 1e-14;
    gamma = 0.9;
    max_iter = 500;
    show_output = 0;  % set to 0 to suppress per-iteration output

    % Preallocate RMSE storage
    rmse_all = zeros(nTrials, 1);

    % Main trial loop
    for trial = 1:nTrials
        % Generate random sensor and anchor locations
        P1 = -100 + 200 * rand(d, m); 
        P2 = -100 + 200 * rand(d, n); 
        P = [P1, P2];  
        P = P - mean(P, 2);  % center the points

        % Compute Gram matrix
        X = P' * P;

        % Compute squared distance matrix
        dist = squareform(pdist(P'));
        D = dist.^2;

        % Extract blocks of D
        E = D(1:m, 1:m);
        F = D(1:m, m+1:end);
        G = D(m+1:end, m+1:end); % not used in this script

        % Determine zeta0 based on F
        zeta0 = 1.2*max(F(:));

        % Add sparse noise to F
        F_corrupted = get_sparse_noise(F, alpha);
        D_corrupted = [E, F_corrupted; F_corrupted', G]; % corrupted distance matrix
        X_corrupted = dist2gram(D_corrupted); % corrupted Gram matrix
        P_corr_estimated = gram_to_points(X_corrupted, d); % estimated points from corrupted Gram matrix
        [rmse_cor, ~, ~] = Compute_RMSE(P', P_corr_estimated);



        % Select a known row of F for the algorithm
        known_row_F = F(known_row_id, :);
        F_corrupted(known_row_id,:) = known_row_F;

        % Apply the AAP_SREDG algorithm
        % [~, P_estimated, ~] = AAP_SREDGRPCA(E, F_corrupted, X, P, d, ...
        %                                 known_row_F, known_row_id, ...
        %                                 zeta0, gamma, max_iter, tol, show_output);

        [~, P_estimated, ~] = SREDG_AAP(E, F_corrupted, F, X, P, d, ...
                                        known_row_F, known_row_id, ...
                                        zeta0, gamma, max_iter, tol, show_output);

        % Compute RMSE between true and estimated sensor locations
        [rmse, ~, ~] = Compute_RMSE(P', P_estimated);
        rmse_all(trial) = rmse;

        fprintf('Trial %d/%d: RMSE = %.4g\n', trial, nTrials, rmse);
        fprintf('Corrupted RMSE = %.4g\n', rmse_cor);
    end

    % Compute and display the mean RMSE
    mean_rmse = mean(rmse_all);
    fprintf('================================\n');
    fprintf('p = %d, m = %d, alpha = %.2f, trial = %d, mean RMSE = %.4g\n', ...
            p, m, alpha, nTrials, mean_rmse);
end

% Helper function: Adds sparse noise to the F block of the distance matrix.
function F_corrupted = get_sparse_noise(F, alpha)
    [m, n] = size(F);
    % Select indices for outliers
    S_supp_idx = randsample(m * n, round(alpha * m * n), false);
    S_range = 1*mean(abs(F(:)));
    % Generate random noise in the range [-S_range, S_range]
    S_temp = 2 * S_range * rand(m, n) - S_range; 
    S_true = zeros(m, n);
    S_true(S_supp_idx) = S_temp(S_supp_idx);  
    F_corrupted = F + S_true;
end


function X = dist2gram(D)
    p = size(D, 1);
    s = zeros(p,1);
    s(1:end) = 1/p;
    J = eye(p) - (ones(p,1)*s');
    X = -0.5*J*D*J';
end

function P = gram_to_points(X, d)
    [V, Lam] = eigs(X, d, 'lm');
    P = V * sqrt(Lam);
end
