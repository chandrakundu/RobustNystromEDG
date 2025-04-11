function [X_estimated, P_estimated, time_counter] = SREDG_AAP(E, F_corrupted, F_star, X_star, P_star, d, known_row_F, known_row_id, zeta0, gamma, max_iter, tol, show_output)

    % SREDG_AAP:  
    % Input:
    %   E: m x m matrix, block E of the distance matrix
    %   F_corrupted: m x n matrix, block F of the distance matrix with added noise
    %   F_star: m x n matrix, block F of the distance matrix without noise
 



    % initialize tracking variables
    timer = zeros(1, max_iter); % time tracking
    time_counter = 0; % total time tracking
    
    % test codes 
    [m, n] = size(F_corrupted);


    % Initialization
    tic;
    S0 = hard_thresholding(F_corrupted, zeta0); % hard thresholding
    Fk = F_corrupted - S0; 
    B0 = operatorB(E, Fk); 
    Bk = projHr(B0, d);  

    init_timer = toc(tic);

    for k = 1:max_iter
        tic;
        Fk_new = operatorA(Bk, E, known_row_F, known_row_id); 


        Rk = F_corrupted - Fk_new; 
        zeta = zeta0 * (gamma^(k-1)); 
        S0 = hard_thresholding(Rk, zeta); 


        Fk_new = F_corrupted - S0;
        Bk_new = operatorB(E, Fk_new);
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
