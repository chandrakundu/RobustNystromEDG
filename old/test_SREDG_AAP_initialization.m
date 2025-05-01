
% This script is for sensor localization synthetic data. 

%% load data
clear; clc;
addpath LIB\
addpath rpca\
trials = 50; % Number of trials
alpha = 0.2; % percentage of outliers
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));
show_output = 0; % Suppress detailed output during trials
max_iter = 200;

p = 500;  % number of points
d = 3;    % dimension of the points
n = p - m; % number of sensors


results = zeros(trials, 1); % Store RMSE for each trial
recovered_count = 0; % Count of successful recoveries

for t = 1:trials
    
    [E, F_corrupted, P, Et, F, G] = generate_data(alpha, m, p, d, 'shuffle'); % generate data
    X = P'*P; % true Gram matrix

    %% Apply AAP_SREDG
    known_row_id = m;
    known_row_F = F(known_row_id,:);  
    tol = 1e-14;
    zeta0 = 1 * max(F(:));
    gamma = 0.9;
    accelerated = true; 
    % [X_estimated, P_estimated, ~] = SREDG_AAP_test_AccAltProj(E, F_corrupted, F, X, P, d, known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol, show_output);
    [X_estimated, P_estimated, ~] = SREDG_AAP_test(E, F_corrupted, F, X, P, d, known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol, show_output);


    %% Compute RMSE
    [rmse, ~, ~] = Compute_RMSE(P', P_estimated);
    results(t) = rmse;
    if rmse < 1e-1
        recovered_count = recovered_count + 1;
    end
    fprintf(' Trial %d: %.4g;', t, rmse);
    if mod(t+1, 8) == 0
        fprintf("\n");
    end
end

% Final statistics
mean_rmse = mean(results);
median_rmse = median(results);
std_rmse = std(results);

fprintf('================================\n');
fprintf('Final Results:\n');
fprintf('Mean RMSE: %.4g\n', mean_rmse);
fprintf('Median RMSE: %.4g\n', median_rmse);
fprintf('Standard Deviation of RMSE: %.4g\n', std_rmse);
fprintf('Number of Successful Recoveries (RMSE < 1e-1): %d/%d\n', recovered_count, trials);
fprintf('================================\n');



%% Visualization
% plot_points(P',P_estimated,m)



%% helper functions



%% SREDG_AAP initialization test
function [X_estimated, P_estimated, time_counter] = SREDG_AAP_test(E, F_corrupted, F_star, X_star, P_star, d, known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol, show_output)

    % SREDG_AAP:  
    % Input:
    %   E: m x m matrix, block E of the distance matrix
    %   F_corrupted: m x n matrix, block F of the distance matrix with added noise
    %   F_star: m x n matrix, block F of the distance matrix without noise
 



    % initialize tracking variables
    timer = zeros(1, max_iter); % time tracking
    
    % test codes 
    [m, n] = size(F_corrupted);


    % Initialization
    tic;
    target_frac = 0.20; 
    % old initialization using hard thresholding
    % S0 = hard_thresholding(F_corrupted, zeta0); % hard thresholding    
    % S0 = threshold_decay(F_corrupted, zeta0, gamma, 1, target_frac);
    % Fk = F_corrupted - S0; 
   
    % New initialization using true value with added Gaussian noise
    % noise_level = 1000; 
    % S0 = zeros(size(F_corrupted)); 
    % Fk = F_star + noise_level * randn(size(F_star)); 

    % experiments 
    % trim_percentile = 99;
    % Fk = robust_initialization(F_corrupted, trim_percentile);

    % r = d+2;
    % para.mu        = 1.1*get_mu_kappa(F_star,r);  
    % para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
    % para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
    % para.trimming  = false;
    % para.tol       = 1e-10;
    % para.gamma     = 0.9;
    % para.max_iter  = 500;
    % para.show_output = 0;
    % para.F_star = F_star;
    % [Fk, ~] = AccAltProj(F_corrupted, d+2, para );

    % accrpca based initialization
    r         = d+2; % rank of the distance matrix
    mu        = 1.1*get_mu_kappa(F_star,r);  % get incohorence parameters of _star
    beta_init = r*sqrt(mu(1)*mu(end))/(sqrt(m*n));
    beta      = r*sqrt(mu(1)*mu(end))/(3*sqrt(m*n));
    zeta      = beta_init * svds(F_corrupted, 1); 
    S0        = wthresh(F_corrupted, 'h', zeta); 
    [U,Sigma,V] = svds(F_corrupted - S0, r);
    Fk = U * Sigma * V'; % initial guess for Fk

    zeta = beta * Sigma(1,1); 
    S = wthresh(F_corrupted - Fk, 'h', zeta); 


    B_corrupted= operatorB(E, F_corrupted);
    zeta0B = max(abs(B_corrupted(:)));
    % B0 = operatorB(E, Fk); 
    % Bk = projHr(B0, d);  

    B_star = compute_B(E, F_star);

    para.mu        = 1.1*get_mu_kappa(B_star,d);  
    para.beta_init = d*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n)); % 
    para.beta      = d*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
    para.trimming  = false;
    para.tol       = 1e-10;
    para.gamma     = 0.9;
    para.max_iter  = 500;
    para.show_output = 0;
    para.B_star = B_star;

    [Bk, ~] = AccAltProj(B_corrupted, d, para );

    
    
    
    init_timer = toc(tic);

    % init errors 
    err_F0 = norm(F_star - Fk, 'fro') / norm(F_star, 'fro');
    err_B0 = norm(B_star - Bk, 'fro') / norm(B_star, 'fro');
    if show_output >= 2
        fprintf('Initial errors: err_F0 = %0.4e, err_B0 = %0.4e\n', err_F0, err_B0);
    end

    

    for k = 1:max_iter
        tic;

        SB = B_corrupted - Bk;
        zetaB = (0.9^k) * zeta0B;
        SB = wthresh(SB, 's', zetaB);
        Bk = B_corrupted - SB; % update Bk

        Fk_new = operatorA(Bk, E, known_row_F, known_row_id); 
        % Z = Fk_new;

        % [Q1,R1] = qr(Z' * U - V * ((Z * V)' * U), 0);
        % [Q2,R2] = qr(Z  * V - U * ( U' * Z  * V), 0);

        % M = [ U'*Z*V, R1'            ;
        %       R2    , zeros(size(R2)); ];
        [U_of_M, Sigma, V_of_M] = svd(Fk_new,'econ');
        % These 2 matrices multiplications can be computed parallelly
        % U = [U, Q2] * U_of_M(:,1:r);
        % V = [V, Q1] * V_of_M(:,1:r);
        % Fk_new = U * Sigma(1:r,1:r) * V';

        Rk = F_corrupted - Fk_new; 
        zeta = beta * (Sigma(r+1,r+1) + (0.9^k)*Sigma(1,1));
        % median_Rk = median(abs(Rk(:)));
        % beta_prime = 1;
        % zeta = beta_prime * median_Rk;
        S = wthresh(Rk, 's', zeta);   

        Fk_new = F_corrupted - S;
        Bk_new = operatorB(E, Fk_new);

        % Project onto tangent space at Bk
        if accelerated
            Bk_new = projTangent(Bk_new, Bk, d); 
        end

        Bk_new = projHr(Bk_new, d);

        timer(k) = toc(tic); % time tracking
        if show_output == 2
            [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);
            error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
            rmse_points = Compute_RMSE(P_star', P_estimated);
            err_FvsF_star = norm(F_star - Fk, 'fro') / norm(F_star, 'fro');
            err_FvsFk = norm(Fk_new - Fk, 'fro') / norm(Fk, 'fro');
            error_Bk = norm(Bk - Bk_new, 'fro') / norm(Bk, 'fro');
            fprintf('i=%d, err_Bk = %0.4e, FvsFs = %0.4e, FvsFk = %0.4e, err_gram = %0.4e, rmse_pt = %0.4e, time = %f seconds\n', k, error_Bk, err_FvsF_star, err_FvsFk, error_gram, rmse_points, timer(k));
        end

        % if error_Bk < tol
        %     break;
        % end

        Bk = Bk_new; % update Bk
        Fk = Fk_new; % update Fk

    end
    time_counter = init_timer + sum(timer);  
    [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);

    if show_output >= 1
        error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
        rmse_points = Compute_RMSE(P_star', P_estimated);
        fprintf('SREDG_AAP: i=%d, error_gram = %.4f, rmse_points = %.4f, time = %f seconds\n', k, error_gram, rmse_points, time_counter);
    end
end


function F_init = robust_initialization(F_corrupted, trim_percentile)
    % robust_initialization: Produce an initialization for the anchor-target block F.
    %
    % F_corrupted: m x n matrix of measured squared distances (corrupted by outliers)
    % trim_percentile: a value between 0 and 100 (e.g., 90) indicating the percentile threshold.
    %
    % For each row, entries above the trim_percentile value (computed from that row)
    % are replaced by the row median.
    %
    % Output:
    %   F_init: the initialized matrix.
    
    [m, n] = size(F_corrupted);
    F_init = F_corrupted; 
    for i = 1:m
        rowVals = F_corrupted(i,:);
        % Compute the target threshold as the trim_percentile for this row.
        thresh = prctile(rowVals, trim_percentile);
        % Compute a robust estimate, such as the median.
        rowMedian = median(rowVals);
        % For entries that exceed the threshold, replace them by the median.
        outlier_idx = rowVals > thresh;
        F_init(i, outlier_idx) = rowMedian;
    end
end

function S = threshold_decay(R, zeta0, gamma, iter, target_frac)
    % threshold_decay applies a decaying threshold with an adaptive percentile check.
    %
    % INPUTS:
    %   R         : The residual matrix.
    %   zeta0     : The initial threshold value.
    %   gamma     : The decay factor (should be in (0,1); e.g., 0.9).
    %   iter      : Current iteration index.
    %   target_frac: Target fraction of entries allowed as outliers (e.g., 0.10 for 10%).
    %
    % OUTPUT:
    %   S : The sparse outlier matrix computed from R.
    
    % Compute the decayed threshold.
    tau_decay = zeta0 * (gamma^(iter-1));
    
    % Also compute a percentile threshold that would flag target_frac entries.
    R_vec = abs(R(:));
    tau_percentile = prctile(R_vec, 100 - (target_frac * 100));
    
    % Use the stricter (lower) threshold.
    tau = max(tau_decay, tau_percentile);
    
    % Apply thresholding.
    S = R .* (abs(R) >= tau_percentile);
end


function [X_estimated, P_estimated, time_counter] = SREDG_AAP_test_AccAltProj(E, F_corrupted, F_star, X_star, P_star, d, known_row_F, known_row_id, zt, gamma, accelerated, max_iter, tol, show_output)
   
    timer = zeros(1, max_iter);
    [m, n] = size(F_corrupted);
    norm_of_F = norm(F_corrupted, 'fro');
    
    %% Set base threshold parameters following AccAltProj
    % Compute beta = 1/(2*nthroot(m*n,4)) and then beta_init = 4*beta.
    beta = 1/(2*nthroot(m*n,4));
    beta_init = 4 * beta;
    
    % Use lansvd if available; otherwise use svds.
    zeta = beta_init * svds(F_corrupted, 1);
    
    % Initialization: obtain initial sparse component via hard thresholding.
    S0 = wthresh(F_corrupted, 'h', zeta);
    % Subtract sparse part to get a preliminary clean estimate.
    Fk = F_corrupted - S0;
    
    % Compute the initial Gram block using your operatorB.
    B0 = operatorB(E, Fk);
    % Project B0 onto the set of rank-d matrices.
    Bk = projHr(B0, d);
    
    init_timer = toc(tic);
    
    % Compute initial relative errors (optional)
    err_F0 = norm(F_star - Fk, 'fro') / norm(F_star, 'fro');
    err_B0 = norm(B0 - Bk, 'fro') / norm(B0, 'fro');
    if show_output >= 2
        fprintf('Initialization: err_F0 = %0.4e, err_B0 = %0.4e\n', err_F0, err_B0);
    end
    
    % Main iterative loop
    for k = 1:max_iter
        tic;
        % Use operatorA to compute the current F estimate from Bk and E.
        Fk_new = operatorA(Bk, E, known_row_F, known_row_id);
        
        % Compute the residual on the observed block.
        Rk = F_corrupted - Fk_new;
        
        % Update threshold: similar to AccAltProj, set
        %   zeta_iter = beta * (largest singular value of (F_corrupted - Fk_new)).
        zeta_iter = beta * svds(F_corrupted - Fk_new, 1);
        
        % Compute the sparse component using hard thresholding.
        S = wthresh(F_corrupted - Fk_new, 'h', zeta_iter);
        
        % Update F estimate by subtracting the sparse component.
        Fk_new = F_corrupted - S;
        
        % Update Gram matrix block via operatorB.
        Bk_new = operatorB(E, Fk_new);
        
        % (Optional) If using acceleration, project Bk_new onto the tangent space
        if accelerated
            Bk_new = projTangent(Bk_new, Bk, d);
        end
        
        % Project onto the set of rank-d matrices.
        Bk_new = projHr(Bk_new, d);
        timer(k) = toc;
        
        if show_output == 2
            [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);
            error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
            rmse_points = Compute_RMSE(P_star', P_estimated);
            fprintf('Iter %d: error_gram = %0.4e, rmse_pt = %0.4e, time = %f sec\n', k, error_gram, rmse_points, timer(k));
        end
        
        % Update current estimates.
        Bk = Bk_new;
        Fk = Fk_new;
        
        % Stopping criterion (optional; uncomment if desired).
        if norm(F_corrupted - Fk, 'fro')/norm_of_F < tol
            break;
        end
    end
    
    time_counter = sum(timer) + init_timer;
    [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);
    
    if show_output >= 1
        error_gram = norm(X_star - X_estimated, 'fro')/norm(X_star, 'fro');
        rmse_points = Compute_RMSE(P_star', P_estimated);
        fprintf('SREDG_AAP (AccAltProj init): iter = %d, error_gram = %.4f, rmse_points = %.4f, total time = %f seconds\n', k, error_gram, rmse_points, time_counter);
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
