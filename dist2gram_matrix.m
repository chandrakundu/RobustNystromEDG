function X = dist2gram_matrix(E, F, tol)
    % dist2gram_matrix - Converts distance matrices to Gram matrices.
    % The structure of D is as follows:
    % D = [E   F
    %      F^T G]
    % We only know E and F, but not G. This function computes the Gram matrix X
    % Syntax: X = dist2gram_matrix(E, F)
    %
    % Inputs:
    %    E - A matrix of size m x m representing distances among anchor nodes.
    %    F - A matrix of size m x n representing distances between anchor and target 
    % nodes.
    %   tol - Optional tolerance parameter for pinv (default: 0.1) 
    % Outputs:
    %    X - A matrix of size (m+n) x (m+n) representing the Gram matrix.

    if nargin < 3
        tol = 0.1;
    end
  
    m = size(E, 1);
    n = size(F, 2);
    p = m+n;

    mean_E_all = mean(E, "all");    % Scalar (mean of all elements of E)
    mean_E_cols = mean(E, 1);       % Row vector, 1 x m (mean of each column of E)
    mean_E_rows = mean(E, 2);       % Column vector, m x 1 (mean of each row of E)
    mean_F_cols = mean(F, 1);       % Row vector, 1 x n (mean of each column of F)

    % calculate Gram matrix for anchor nodes
    mean_E_cols_exp = repmat(mean_E_cols, m, 1); % Repeat mean_E_cols for m rows (m x m)
    mean_E_rows_exp = repmat(mean_E_rows, 1, m); % Repeat mean_E_rows for m columns (m x m)

    Gram_A = -0.5 * (E - mean_E_cols_exp - mean_E_rows_exp + mean_E_all);

    % calculate Gram matrix for anchor-target nodes
    mean_E_rows_for_F = repmat(mean_E_rows, 1, n); % Expand mean_E_rows for F (m x n)
    mean_F_cols_exp = repmat(mean_F_cols, m, 1);   % Expand mean_F_cols for matrix subtraction in Gram_B (m x n)

    Gram_B = -0.5 * (F - mean_F_cols_exp - mean_E_rows_for_F + mean_E_all);

    % s = zeros(p,1);
    % s(1:m) = 1/m;
    % J = eye(p) - (ones(p,1)*s');
    % % D = [E F;F' zeros(n,n)];
    % D = [E F;F' F'*(pinv(E,tol))*F];
    % X = -0.5 * J*D*J';

    X = [Gram_A Gram_B; Gram_B' Gram_B'*(pinv(Gram_A,tol))*Gram_B];
end
