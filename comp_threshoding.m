%% test_all_thresholds.m
% This script tests several thresholding operators for the SREDG_AAP algorithm.
% We compare seven methods:
%    1. Hard thresholding (global)
%    2. Soft thresholding (global)
%    3. Percentile thresholding (global)
%    4. Weight decay thresholding (global)
%    5. MAD-based thresholding (global)
%    6. Row-wise percentile thresholding (local per row)
%    7. Iteratively reweighted soft thresholding (global)
%
% For each method, we try several parameter combinations.
% We run 50 trials for each configuration (using m=30 anchors and alpha=0.2).
% For each setting, we record the mean RMSE, median RMSE, std, and number of recoveries.
% Results are saved to a text file.

clear; clc;
addpath LIB\
addpath rpca\

% Fixed experiment parameters
trials = 50;        % number of trials per configuration
alpha = 0.2;        % percentage of outliers (20%)
m = 30;             % number of anchors
p = 500;            % total number of points
d = 3;              % dimension of the points
show_output = 0;    % 0 for no detailed trial output

% SREDG_AAP algorithm parameters (common)
tol     = 1e-14;
zeta0   = max(ones(m, p-m)); % we use 1 as a scale (will be tuned per method)
gamma_val = 0.9;     % decay factor (for methods that use weight decay)
max_iter= 200;
accelerated = true;

% Define list of thresholding methods and parameter settings.
% Each entry in "operatorList" is a struct with:
%   - name: string name of the operator.
%   - fn: function handle (expects call as: S = fn(R, params, iter))
%   - paramNames: cell array (for record-keeping)
%   - paramGrid: cell array of arrays (each array is a list of values to try for that parameter).
%
operatorList = {};

%% 1. Hard thresholding (global)
operatorList{end+1} = struct(...
    'name', 'hard', ...
    'fn', @threshold_hard, ...
    'paramNames', {{'tau'}}, ...
    'paramGrid', {{[0.005, 0.01, 0.02]}});

%% 2. Soft thresholding (global)
operatorList{end+1} = struct(...
    'name', 'soft', ...
    'fn', @threshold_soft, ...
    'paramNames', {{'tau'}}, ...
    'paramGrid', {{[0.005, 0.01, 0.02]}});

%% 3. Percentile thresholding (global)
operatorList{end+1} = struct(...
    'name', 'percentile', ...
    'fn', @threshold_percentile, ...
    'paramNames', {{'perc'}}, ...
    'paramGrid', {{[5, 10, 15]}});

%% 4. Weight decay thresholding (global)
operatorList{end+1} = struct(...
    'name', 'weight_decay', ...
    'fn', @threshold_weight_decay, ...
    'paramNames', {{'tau0','gamma'}}, ...
    'paramGrid', {{[0.005, 0.01, 0.02], gamma_val}}); % gamma fixed at 0.9

%% 5. MAD-based thresholding (global)
operatorList{end+1} = struct(...
    'name', 'mad', ...
    'fn', @threshold_mad, ...
    'paramNames', {{'multiplier'}}, ...
    'paramGrid', {{[3, 5, 7]}});

%% 6. Row-wise percentile thresholding (local)
operatorList{end+1} = struct(...
    'name', 'row_percentile', ...
    'fn', @threshold_row_percentile, ...
    'paramNames', {{'perc'}}, ...
    'paramGrid', {{[5, 10, 15]}});

%% 7. Iteratively reweighted soft thresholding (global)
operatorList{end+1} = struct(...
    'name', 'iter_reweighted_soft', ...
    'fn', @threshold_iter_reweighted_soft, ...
    'paramNames', {{'tau'}}, ...
    'paramGrid', {{[0.005, 0.01, 0.02]}});

% Open result output file.
output_filename = 'threshold_operator_results.txt';
fid = fopen(output_filename, 'w');
fprintf(fid, 'Thresholding Method Results for m = %d, alpha = %.2f over %d trials\n\n', m, alpha, trials);

% Loop over each threshold operator
for op = 1:length(operatorList)
    opStruct = operatorList{op};
    baseName = opStruct.name;
    fn_handle = opStruct.fn;
    paramNames = opStruct.paramNames;
    paramGrid = opStruct.paramGrid;
    
    % Create all combinations of parameters.
    % For a single parameter, this is trivial.
    % For more than one, use ndgrid.
    numParams = length(paramNames);
    if numParams == 1
        paramCombos = num2cell(paramGrid{1});
    else
        [paramGrids{1:numParams}] = ndgrid(paramGrid{:});
        paramCombos = cell(numel(paramGrids{1}), numParams);
        for i = 1:numParams
            paramCombos(:, i) = num2cell(paramGrids{i}(:));
        end
    end
    
    % Loop over each parameter combination
    for comboIdx = 1:size(paramCombos,1)
        % Build parameter struct for this combination.
        params = struct();
        for pIdx = 1:numParams
            params.(paramNames{pIdx}) = paramCombos{comboIdx, pIdx};
        end
        
        configName = sprintf('%s_%s', baseName, jsonencode(params));
        fprintf(fid, 'Method: %s\nParameters: %s\n', baseName, jsonencode(params));
        fprintf('Testing %s\n', configName);
        
        % Initialize arrays to collect trial statistics.
        rmseList = zeros(trials,1);
        recovered_count = 0;
        
        % Run trials for this configuration.
        for t = 1:trials
            % Generate synthetic sensor localization data.
            [E, F_corrupted, P, ~, F_true, ~] = generate_data(alpha, m, p, d, 'shuffle');
            X_true = P' * P;
            known_row_id = m;
            known_row_F = F_true(known_row_id, :);
            
            % Run SREDG_AAP_test with the chosen threshold operator.
            [X_estimated, P_estimated, ~] = SREDG_AAP_test(E, F_corrupted, F_true, X_true, P, d, ...
                known_row_F, known_row_id, zeta0, gamma_val, accelerated, max_iter, tol, show_output, ...
                fn_handle, params);
            
            % Compute RMSE using your Compute_RMSE function.
            [rmse, ~, ~] = Compute_RMSE(P', P_estimated);
            rmseList(t) = rmse;
            if rmse < 1e-1
                recovered_count = recovered_count + 1;
            end
        end % trials loop
        
        mean_rmse = mean(rmseList);
        median_rmse = median(rmseList);
        std_rmse = std(rmseList);
        
        % Write configuration result to file.
        fprintf(fid, 'Mean RMSE: %.4g\n', mean_rmse);
        fprintf(fid, 'Median RMSE: %.4g\n', median_rmse);
        fprintf(fid, 'Std RMSE: %.4g\n', std_rmse);
        fprintf(fid, 'Recovered (RMSE < 1e-1): %d/%d\n', recovered_count, trials);
        fprintf(fid, '------------------------------------------\n\n');
        fprintf('Done %s: Mean=%.4g, Recovery=%d/%d\n', configName, mean_rmse, recovered_count, trials);
    end
end

fclose(fid);
fprintf('Results written to %s\n', output_filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Thresholding Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function S = threshold_hard(R, params, iter)
    % Hard thresholding: flag entries with |R| >= tau.
    tau = params.tau;
    S = R .* (abs(R) >= tau);
end

function S = threshold_soft(R, params, iter)
    % Soft thresholding: shrink residuals.
    tau = params.tau;
    S = sign(R) .* max(abs(R) - tau, 0);
end

function S = threshold_percentile(R, params, iter)
    % Percentile thresholding: flag only the top perc% largest values.
    perc = params.perc;
    R_vec = abs(R(:));
    tau = prctile(R_vec, 100 - perc);
    S = R .* (abs(R) >= tau);
end

function S = threshold_weight_decay(R, params, iter)
    % Weight decay thresholding: tau decays as tau0 * gamma^(iter-1)
    tau0 = params.tau0;
    gamma_val = params.gamma;
    tau = tau0 * (gamma_val^(iter-1));
    S = R .* (abs(R) >= tau);
end

function S = threshold_mad(R, params, iter)
    % MAD-based thresholding: flag entries > multiplier * MAD.
    multiplier = params.multiplier;
    med_R = median(R(:));
    mad_R = median(abs(R(:) - med_R));
    tau = multiplier * mad_R;
    S = R .* (abs(R) >= tau);
end

function S = threshold_row_percentile(R, params, iter)
    % Row-wise percentile thresholding: each row uses its own threshold.
    perc = params.perc;
    S = zeros(size(R));
    for i = 1:size(R,1)
        rowVals = abs(R(i,:));
        tau = prctile(rowVals, 100 - perc);
        S(i,:) = R(i,:) .* (rowVals >= tau);
    end
end

function S = threshold_iter_reweighted_soft(R, params, iter)
    % Iteratively reweighted soft thresholding:
    % Compute weights: w = 1/(1+(|R|/tau)^2), then outlier estimate S = R .* (1 - w).
    tau = params.tau;
    w = 1 ./ (1 + (abs(R)/tau).^2);
    S = R .* (1 - w);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modified SREDG_AAP_test Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_estimated, P_estimated, time_counter] = SREDG_AAP_test(E, F_corrupted, F_star, X_star, P_star, d, ...
    known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol, show_output, threshold_fn, threshold_params)
% SREDG_AAP_test: Modified to accept a thresholding function handle and parameters.
%
% Inputs:
%   E, F_corrupted, F_star, X_star, P_star, d, known_row_F, known_row_id,
%   zeta0, gamma, accelerated, max_iter, tol, show_output: as usual.
%   threshold_fn: function handle for thresholding.
%   threshold_params: parameters struct for threshold_fn.
%
% Outputs:
%   X_estimated, P_estimated, time_counter.

    timer = zeros(1, max_iter); 
    [m, n] = size(F_corrupted);

    tic;
    % Initialization: use threshold_fn on F_corrupted at iteration 1.
    noise_level = 1000; 
    Fk = F_star + noise_level * randn(size(F_star)); 
   
    B0 = operatorB(E, Fk); 
    Bk = projHr(B0, d);  
    init_timer = toc(tic);

    for k = 1:max_iter
        tic;
        Fk_new = operatorA(Bk, E, known_row_F, known_row_id);
        Rk = F_corrupted - Fk_new;
        
        S = threshold_fn(Rk, threshold_params, k);
        
        Fk_new = F_corrupted - S;
        Bk_new = operatorB(E, Fk_new);
        
        if accelerated
            Bk_new = projTangent(Bk_new, Bk, d);
        end
        
        Bk_new = projHr(Bk_new, d);
        timer(k) = toc(tic);
        
        if show_output == 2
            [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);
            error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
            rmse_points = Compute_RMSE(P_star', P_estimated);
            fprintf('i=%d, error_gram = %0.4e, rmse_pt = %0.4e, time = %f seconds\n', k, error_gram, rmse_points, timer(k));
        end

        Bk = Bk_new;
        Fk = Fk_new;
    end
    time_counter = init_timer + sum(timer);  
    [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);

    if show_output >= 1
        error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
        rmse_points = Compute_RMSE(P_star', P_estimated);
        fprintf('SREDG_AAP: i=%d, error_gram = %.4f, rmse_points = %.4f, time = %f seconds\n', k, error_gram, rmse_points, time_counter);
    end
end


function F = operatorA(B, E, known_row_F, k)
    F = compute_F(B, E, known_row_F, k);
end

function B = operatorB(E, F)
    B = compute_B(E, F);
end

function S = hard_thresholding(F, zeta)
    % HARD_THRESHOLDING Performs hard thresholding on the matrix F
    S = F .* (abs(F) > zeta);
end

function Xr = projHr(X,r)
    % PROJHR Projects the matrix X onto the set of matrices with rank r
    % using Singular Value Decomposition (SVD)
    [U, S, V] = svd(X, 'econ');
    S = diag(S);
    S(r+1:end) = 0; % Set all singular values after r to zero
    Xr = U * diag(S) * V'; % Reconstruct the matrix with rank r
end

function Z_proj = projTangent(Z, Bk, d)
    % projTangent:  P_{T^k}(Z) = Uk*Uk'^T Z + Z Vk*Vk'^T - Uk*Uk'^T Z Vk*Vk'^T
    % Uk, Vk are the top-r left and right singular vectors of B^k
    % Projects Z onto the tangent space of rank-r manifold at B^k
        [Uk, ~, Vk] = svds(Bk, d);
        UkUt = Uk*(Uk');
        VkVt = Vk*(Vk');
        Z_proj = UkUt*Z + Z*VkVt - UkUt*Z*VkVt;
end


function A = compute_A(E)
    % compute_A Computes block A of the Gram matrix from E block of distance matrices

    % Dimension of E 
    m = size(E, 1);

    % Vector and matrix of ones
    ones_m = ones(m, 1);
    ones_mm = (1/m) * (ones_m * ones_m');

    A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
end

function B = compute_B(E, F)
    % COMPUTE_B Computes block B of the Gram matrix from E and F blocks of distance matrices

    % Dimensions of E and F
    m = size(E, 1);
    n = size(F, 2);

    % Vector of ones
    ones_m = ones(m, 1);
    ones_n = ones(n, 1);

    ones_mm = (1/m) * (ones_m * ones_m');
    ones_mn = (1/m) * (ones_m * ones_n');

    B = -0.5 * (F - ones_mm * F - E * ones_mn + m*mean(E(:)) * ones_mn);
end


function F = compute_F(B, E, known_row_F, k)
    % COMPUTE_F Computes the F block of the Gram matrix from B, E, and a known row of F
    %
    % Inputs:
    % B: m x n matrix, block B of the Gram matrix
    % E: m x m matrix, block E of the distance matrix
    % known_row_F: 1 x n vector, known row of F
    % k: integer, index of the known row
    %
    % Output:
    % F: m x n matrix, the F block of the Gram matrix

    % Dimensions of B
    [m, n] = size(B);

    rowSumE = sum(E, 2); 
    diffRowSumE = rowSumE - rowSumE(k);

    diffB = B - repmat(B(k,:), m, 1);

    FrowPart = repmat(known_row_F, m, 1);

    F = FrowPart - 2 * diffB + (1/m) * diffRowSumE * ones(1, n);
end


% function for comparing results and display 
function [X, P] = get_current_estimates(Bk, E, d)
    X = EB_to_gram(E, Bk);
    P = gram_to_points(X, d);
    P_centered = P - mean(P, 1);
    X = P_centered*P_centered';
end


function L = EF_to_gram(E, F)
    % EF_to_gram:  Compute the Gram matrix from E and F blocks of distance matrices
    %   L = [A B; B' C]
    A = compute_A(E, F);
    B = compute_B(E, F);
    L = AB_to_gram(A, B);
end

function L = EB_to_gram(E, B)
    % EB_to_gram:  Compute the Gram matrix from E block of distance matrix and B block of gram matrix
    %   L = [A B; B' C]
    A = compute_A(E);
    L = AB_to_gram(A, B);
end

function L = AB_to_gram(A, B)
    % AB_to_gram:  Compute the Gram matrix from A and B blocks of gram matrix
    %   L = [A B; B' C]
    C = B'*pinv(A, 0.01)*B;
    L = [A B; B' C];
    L = fix_gram_matrix(L);
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
