function X = dist2gram(E, F, nystrom_on, tol)

    switch nargin
        case 1
            D = E;
            nystrom_on = "none";
            tol = 0.01;
        case 2
            nystrom_on = "gram";
            tol = 0.01;
        case 3
            tol = 0.01;
    end

    if nystrom_on == "none"
        X = dist2gram_nystrom_on_none(D);
    elseif nystrom_on == "dist"
        X = dist2gram_nystrom_on_distance(E,F,tol);
    elseif nystrom_on == "gram"
        X = dist2gram_nystrom_on_gram(E,F,tol);
    end

    % deal with negative eigenvalues of X
    X = fix_negative_eigenvalues(X);

end

function X = dist2gram_nystrom_on_none(D)
    p = size(D, 1);
    s = zeros(p,1);
    s(1:end) = 1/p;
    J = eye(p) - (ones(p,1)*s');
    X = -0.5*J*D*J';
end


function X = dist2gram_nystrom_on_distance(E,F,tol)
    m = size(E, 1);
    n = size(F, 2);
    p = m+n;
    s = zeros(p,1);
    s(1:m) = 1/m;
    J = eye(p) - (ones(p,1)*s');
    D = [E F;F' F'*(pinv(E,tol))*F];
    X = -0.5 * J*D*J';
end

function X = dist2gram_nystrom_on_gram(E,F,tol)
        if nargin < 3
            tol = 0.01;
        end
      
        m = size(E, 1);
        n = size(F, 2);
        % p = m+n;
    
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

        X = [Gram_A Gram_B; Gram_B' Gram_B'*(pinv(Gram_A,tol))*Gram_B];
    
        % s = zeros(p,1);
        % s(1:m) = 1/m;
        % J = eye(p) - (ones(p,1)*s');
        % D = [E F;F' zeros(n,n)];
        % X = -0.5 * J*D*J';    
    end



function X_new = fix_negative_eigenvalues(X, tolerance)
    % FIX_NEGATIVE_EIGENVALUES Adjusts small negative eigenvalues to zero and reconstructs the matrix.
    
    if nargin < 2
        tolerance = 0; % Default tolerance
    end
    
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
