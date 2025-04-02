function [X,P,time_counter] = SREDG(E,F,r,RPCA, para)
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

    % clean the distance matrix between anchor and target nodes
    if isfield(para, 'show_output')
        show_output = para.show_output;
    else
        show_output = 2;
    end

    time_counter = 0;
    tstart = tic;
    [F_hat, ~] = RPCA(F, r, para );

    % compute A and B
    [A, B] = compute_AB(E, F_hat);

    % compute the Gram matrix
    C = B'*pinv(A, 0.01)*B;
    X = [A B; B' C];

    tEnd = toc(tstart);
    time_counter = time_counter + tEnd;

    if show_output >= 1
        fprintf('SREDG: time = %f seconds\n', time_counter);
    end

    % fix the negative eigenvalues and ensure symmetry
    X = fix_gram_matrix(X);
    P = gram_to_points(X, r-2);
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


function Xmat = gram_to_points(L, r)    
        [V, Lam] = eigs(L, r, 'lm');
        Xmat = V*sqrt(Lam);
    end
