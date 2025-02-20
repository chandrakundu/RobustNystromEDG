function X = SREDG_rpcaB(E,F,r,RPCA, para)
    % SREDG Sparse Robust Euclidean Distance Geometry algorithm for distance matrix completion
    %
    % Input:
    % E: m x m matrix, the distance matrix of anchor nodes
    % F: m x n matrix, the distance matrix between anchor and target nodes
    % r: the rank of the distance matrix (typically dimension + 2)
    % RPCA: the function handle of Robust PCA algorithm
    % para: the parameter for RPCA  (e.g., AccAltProj)
    %
    % Output:
    % X: (m+n) x (m+n) matrix, the Gram matrix of the distance matrix

    % compute A and B
    [A, B] = compute_AB(E, F);

    % clean the distance matrix between anchor and target nodes
    [B_hat, ~] = RPCA(B, r, para );

    % compute the Gram matrix
    C = B_hat'*pinv(A, 0.01)*B_hat;
    X = [A B_hat; B_hat' C];

    % fix the negative eigenvalues and ensure symmetry
    X = fix_gram_matrix(X);
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
